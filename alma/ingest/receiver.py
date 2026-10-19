"""The UDP receiver: a thread with its own socket that decodes telegrams from iba and puts them in the cache.

Each datagram has one of four outcomes, and each outcome is a counter in stats():

    accepted   fits the format and has been stored
    blocked    the source address is not in allowed_sources (None allows every address)
    malformed  does not fit the format (length, check word, NaN or outside min/max)
    dropped    the format is correct, but the cache refused the row (timestamp not increasing)

Per datagram the receiver does nothing heavier than a filter, a struct call and a cache append: it runs
for every packet, including from whoever floods the port. Refused telegrams are counted and logged at most
once per minute, without packet contents.

The timestamp is the receive time (clock). Heartbeat detection and an exact loss measurement from a counter
in the telegram do not exist yet: stats() reports frozen=False and loss_percent=None, and UdpSource then
estimates the loss from the arrival times.
"""

import ipaddress
import logging
import socket
import threading
import time
from collections.abc import Callable, Sequence

from alma.ingest.cache import SampleCache
from alma.ingest.format import MAX_PACKET_BYTES, PacketError, PacketFormat

__all__ = ["UdpReceiver"]

log = logging.getLogger("alma.udp")

POLL_S = 0.2  # how quickly the thread notices it has to stop
LOG_EVERY_S = 60  # at most one log line per this many seconds for refused telegrams
RCVBUF_BYTES = 1 << 20  # generous, so a short stall of the thread does not cost telegrams


class UdpReceiver:
    def __init__(
        self,
        fmt: PacketFormat,
        cache: SampleCache,
        *,
        bind: str = "0.0.0.0",
        port: int = 5005,
        allowed_sources: Sequence[str] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._fmt = fmt
        self._cache = cache
        self._bind = bind
        self._port = port
        # None: every source address; an empty list: nobody
        self._networks = (
            None if allowed_sources is None else tuple(ipaddress.ip_network(s, strict=False) for s in allowed_sources)
        )
        self._clock = clock

        self._lock = threading.Lock()  # counters and log counter; the thread writes, stats() reads
        self._counts = {"received": 0, "accepted": 0, "dropped": 0, "blocked": 0, "malformed": 0}
        self._last_accepted: float | None = None
        self._unlogged = 0
        self._last_log: float | None = None  # time.monotonic(), independent of the (sometimes test-driven) clock

        self._stop = threading.Event()
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        """The port being listened on; after start() this is the system-chosen port, also for port=0."""
        return self._port

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Open the socket and start the thread. If opening fails (port in use), raise a RuntimeError."""
        if self.running:
            return
        family = socket.AF_INET6 if ipaddress.ip_address(self._bind).version == 6 else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_DGRAM)
        try:
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RCVBUF_BYTES)
            except OSError:
                pass  # a smaller buffer is no reason not to start
            sock.bind((self._bind, self._port))  # deliberately without SO_REUSEADDR: a port in use must be noticed
        except OSError as err:
            sock.close()
            raise RuntimeError(
                f"Cannot open UDP port {self._port} on {self._bind} ({err.strerror or err}). Is another process "
                "already running on that port, such as alma.udp_check or a second dashboard?"
            ) from err
        sock.settimeout(POLL_S)
        self._port = sock.getsockname()[1]
        self._stop = threading.Event()
        self._sock = sock
        self._thread = threading.Thread(target=self._run, args=(sock, self._stop), name="udp-receiver", daemon=True)
        self._thread.start()
        log.info("UDP receiver listening on %s:%d", self._bind, self._port)

    def stop(self) -> None:
        """Stop the thread and release the port."""
        thread, sock = self._thread, self._sock
        if thread is None or sock is None:
            return
        self._stop.set()
        thread.join(timeout=POLL_S + 2)
        sock.close()
        self._thread = self._sock = None

    def stats(self) -> dict:
        """Counters at this moment. last_packet_age_s is the age of the last accepted telegram."""
        with self._lock:
            counts = dict(self._counts)
            last = self._last_accepted
        age = None if last is None else round(max(0.0, self._clock() - last), 1)
        return {**counts, "last_packet_age_s": age, "frozen": False, "loss_percent": None}

    # ---- the receive loop ------------------------------------------------------------------------------

    def _run(self, sock: socket.socket, stop: threading.Event) -> None:
        buffer = bytearray(MAX_PACKET_BYTES + 1)  # a datagram always fits, so nothing is truncated
        view = memoryview(buffer)
        while not stop.is_set():
            try:
                size, address = sock.recvfrom_into(buffer)
            except socket.timeout:
                continue
            except OSError as err:
                if stop.is_set():
                    break
                self._note(f"socket error while receiving: {err.strerror or type(err).__name__}")
                time.sleep(POLL_S)  # do not keep looping on the same error
                continue
            host = address[0]
            try:
                self._handle(view[:size], host)
            except Exception as err:  # the thread must never stop because of a single telegram
                self._record("malformed", host, f"unexpected error while processing: {type(err).__name__}")

    def _handle(self, data: memoryview, host: str) -> None:
        if not self._allowed(host):
            self._record("blocked", host, "source address is not in ALMA_UDP_ALLOWED_SOURCES")
            return
        now = self._clock()
        try:
            values = self._fmt.decode(data)
        except PacketError as err:
            self._record("malformed", host, str(err))
            return
        if not self._cache.append(now, values):
            self._record("dropped", host, "cache refused the row (timestamp not increasing or invalid value)")
            return
        with self._lock:
            self._counts["received"] += 1
            self._counts["accepted"] += 1
            self._last_accepted = now
            first = self._counts["accepted"] == 1
        if first:
            log.info("First telegram received from %s", host)

    def _allowed(self, host: str) -> bool:
        if self._networks is None:
            return True
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return False
        ip = getattr(ip, "ipv4_mapped", None) or ip  # an IPv4 sender on a dual-stack socket
        return any(ip in network for network in self._networks)

    def _record(self, outcome: str, host: str, reason: str) -> None:
        """Count a refused telegram and log it at most once per LOG_EVERY_S, without contents."""
        with self._lock:
            self._counts["received"] += 1
            self._counts[outcome] += 1
            count = self._claim_log_slot()
        if count:
            log.warning("%d telegram(s) refused (last: %s, source %s): %s", count, outcome, host, reason)

    def _note(self, reason: str) -> None:
        with self._lock:
            count = self._claim_log_slot()
        if count:
            log.warning("UDP receiver: %s", reason)

    def _claim_log_slot(self) -> int:
        """Under the lock: how many reports may be logged now (0 = do not log yet)."""
        self._unlogged += 1
        now = time.monotonic()
        if self._last_log is not None and now - self._last_log < LOG_EVERY_S:
            return 0
        count, self._unlogged = self._unlogged, 0
        self._last_log = now
        return count
