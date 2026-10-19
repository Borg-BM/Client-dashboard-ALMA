"""De UDP-ontvanger: een thread met een eigen socket die telegrammen van iba decodeert en in de cache zet.

Elk datagram krijgt een van vier uitkomsten, en elke uitkomst is een teller in stats():

    accepted   past bij het formaat en is opgeslagen
    blocked    het bronadres staat niet in allowed_sources (None laat elk adres toe)
    malformed  past niet bij het formaat (lengte, controlewoord, NaN of buiten min/max)
    dropped    het formaat klopt, maar de cache weigerde de rij (tijdstip niet oplopend)

De ontvanger doet per datagram niets zwaarders dan een filter, een struct-aanroep en een cache-append: het
draait voor elk pakket, ook van wie de poort bestookt. Geweigerde telegrammen worden geteld en hooguit eens
per minuut gelogd, zonder pakketinhoud.

De tijdstempel is de ontvangsttijd (clock). Heartbeat-detectie en een exacte verliesmeting uit een teller
in het telegram zijn er nog niet: stats() meldt frozen=False en loss_percent=None, en UdpSource schat het
verlies dan uit de aankomsttijden.
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

POLL_S = 0.2  # hoe snel de thread merkt dat hij moet stoppen
LOG_EVERY_S = 60  # hooguit een logregel per zoveel seconden voor geweigerde telegrammen
RCVBUF_BYTES = 1 << 20  # ruim, zodat een korte stilstand van de thread geen telegrammen kost


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
        # None: elk bronadres; een lege lijst: niemand
        self._networks = (
            None if allowed_sources is None else tuple(ipaddress.ip_network(s, strict=False) for s in allowed_sources)
        )
        self._clock = clock

        self._lock = threading.Lock()  # tellers en logteller; de thread schrijft, stats() leest
        self._counts = {"received": 0, "accepted": 0, "dropped": 0, "blocked": 0, "malformed": 0}
        self._last_accepted: float | None = None
        self._unlogged = 0
        self._last_log: float | None = None  # time.monotonic(), los van de (soms bediende) clock

        self._stop = threading.Event()
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        """De poort waarop wordt geluisterd; na start() ook bij port=0 de door het systeem gekozen poort."""
        return self._port

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Open de socket en start de thread. Mislukt het openen (poort bezet), dan een RuntimeError."""
        if self.running:
            return
        family = socket.AF_INET6 if ipaddress.ip_address(self._bind).version == 6 else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_DGRAM)
        try:
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RCVBUF_BYTES)
            except OSError:
                pass  # een kleinere buffer is geen reden om niet te starten
            sock.bind((self._bind, self._port))  # bewust zonder SO_REUSEADDR: een bezette poort moet opvallen
        except OSError as err:
            sock.close()
            raise RuntimeError(
                f"Kan UDP-poort {self._port} op {self._bind} niet openen ({err.strerror or err}). Draait er al "
                "een ander proces op die poort, zoals alma.udp_check of een tweede dashboard?"
            ) from err
        sock.settimeout(POLL_S)
        self._port = sock.getsockname()[1]
        self._stop = threading.Event()
        self._sock = sock
        self._thread = threading.Thread(target=self._run, args=(sock, self._stop), name="udp-receiver", daemon=True)
        self._thread.start()
        log.info("UDP-ontvanger luistert op %s:%d", self._bind, self._port)

    def stop(self) -> None:
        """Stop de thread en geef de poort vrij."""
        thread, sock = self._thread, self._sock
        if thread is None or sock is None:
            return
        self._stop.set()
        thread.join(timeout=POLL_S + 2)
        sock.close()
        self._thread = self._sock = None

    def stats(self) -> dict:
        """Tellers van dit ene moment. last_packet_age_s is de leeftijd van het laatste geaccepteerde telegram."""
        with self._lock:
            counts = dict(self._counts)
            last = self._last_accepted
        age = None if last is None else round(max(0.0, self._clock() - last), 1)
        return {**counts, "last_packet_age_s": age, "frozen": False, "loss_percent": None}

    # ---- de ontvangstlus -------------------------------------------------------------------------------

    def _run(self, sock: socket.socket, stop: threading.Event) -> None:
        buffer = bytearray(MAX_PACKET_BYTES + 1)  # een datagram past er altijd in, dus er wordt niets afgekapt
        view = memoryview(buffer)
        while not stop.is_set():
            try:
                size, address = sock.recvfrom_into(buffer)
            except socket.timeout:
                continue
            except OSError as err:
                if stop.is_set():
                    break
                self._note(f"socketfout bij ontvangen: {err.strerror or type(err).__name__}")
                time.sleep(POLL_S)  # niet in een lus op dezelfde fout blijven draaien
                continue
            host = address[0]
            try:
                self._handle(view[:size], host)
            except Exception as err:  # de thread mag nooit stoppen door een enkel telegram
                self._record("malformed", host, f"onverwachte fout bij verwerken: {type(err).__name__}")

    def _handle(self, data: memoryview, host: str) -> None:
        if not self._allowed(host):
            self._record("blocked", host, "bronadres staat niet in ALMA_UDP_ALLOWED_SOURCES")
            return
        now = self._clock()
        try:
            values = self._fmt.decode(data)
        except PacketError as err:
            self._record("malformed", host, str(err))
            return
        if not self._cache.append(now, values):
            self._record("dropped", host, "cache weigerde de rij (tijdstip niet oplopend of ongeldige waarde)")
            return
        with self._lock:
            self._counts["received"] += 1
            self._counts["accepted"] += 1
            self._last_accepted = now
            first = self._counts["accepted"] == 1
        if first:
            log.info("Eerste telegram ontvangen van %s", host)

    def _allowed(self, host: str) -> bool:
        if self._networks is None:
            return True
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return False
        ip = getattr(ip, "ipv4_mapped", None) or ip  # een IPv4-afzender op een dual-stack socket
        return any(ip in network for network in self._networks)

    def _record(self, outcome: str, host: str, reason: str) -> None:
        """Tel een geweigerd telegram en log het hooguit eens per LOG_EVERY_S, zonder inhoud."""
        with self._lock:
            self._counts["received"] += 1
            self._counts[outcome] += 1
            count = self._claim_log_slot()
        if count:
            log.warning("%d telegram(men) geweigerd (laatste: %s, bron %s): %s", count, outcome, host, reason)

    def _note(self, reason: str) -> None:
        with self._lock:
            count = self._claim_log_slot()
        if count:
            log.warning("UDP-ontvanger: %s", reason)

    def _claim_log_slot(self) -> int:
        """Onder het slot: hoeveel meldingen er nu gelogd mogen worden (0 = nog niet loggen)."""
        self._unlogged += 1
        now = time.monotonic()
        if self._last_log is not None and now - self._last_log < LOG_EVERY_S:
            return 0
        count, self._unlogged = self._unlogged, 0
        self._last_log = now
        return count
