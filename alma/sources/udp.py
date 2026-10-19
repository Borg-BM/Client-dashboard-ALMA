"""Live data source: UDP telegrams from iba, the last few minutes held in an in-memory cache.

The ibaHD server does not deliver in real time. iba (ibaPDA, Generic UDP output module) therefore sends
a telegram with all signals to this port every cycle. The receiver decodes them according to
udp_format.json and writes them into a ring buffer; read() reads the requested window from it, so that
analysis and API do not notice that the data arrives live.

One process: the receiver is a thread with its own socket. Multiple uvicorn workers would each want to
open the same port and split the packets between them; so run this with a single worker.
"""

import ipaddress
import time
from collections.abc import Callable, Sequence
from datetime import datetime

import pandas as pd

from alma.analysis import epoch_seconds
from alma.config import SIGNALS, Settings
from alma.ingest.cache import SampleCache
from alma.ingest.format import PacketFormat
from alma.ingest.linkquality import estimate_loss_percent
from alma.ingest.receiver import UdpReceiver

# Mode number: when a window is shrunk the maximum counts, so that a short period is preserved and no
# average of mode numbers arises.
FLAG_SIGNALS = ("system_mode",)

LOSS_WINDOW_S = 60  # the look-back over which the loss is estimated


class UdpSource:
    name = "udp"

    def __init__(
        self,
        fmt: PacketFormat,
        *,
        bind: str = "0.0.0.0",
        port: int = 5005,
        allowed_sources: Sequence[str] | None = None,
        cache_minutes: int = 10,
        max_samples: int = 200_000,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.retention_minutes = cache_minutes  # the frontend offers no longer periods
        self._clock = clock
        self.cache = SampleCache(
            SIGNALS, retention_s=cache_minutes * 60, max_samples=max_samples, flags=FLAG_SIGNALS, clock=clock
        )
        self.receiver = UdpReceiver(
            fmt, self.cache, bind=bind, port=port, allowed_sources=allowed_sources, clock=clock
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> "UdpSource":
        fmt = PacketFormat.load(settings.udp_format, required_signals=SIGNALS)
        if settings.udp_allow_any_source and not fmt.checks and not ipaddress.ip_address(settings.udp_bind).is_loopback:
            # The ship connects via Starlink (changing address), so a source filter is often not usable there. In that
            # case the telegram itself must carry at least a key word, otherwise anyone on the internet can write to it.
            raise ValueError(
                "ALMA_UDP_ALLOW_ANY_SOURCE=true on a non-local address requires a 'checks' key word in "
                f"{settings.udp_format.name}; otherwise anyone who can reach the port can inject measurements. "
                "Preferably use a VPN tunnel with ALMA_UDP_ALLOWED_SOURCES (see the README)"
            )
        return cls(
            fmt,
            bind=settings.udp_bind,
            port=settings.udp_port,
            allowed_sources=None if settings.udp_allow_any_source else settings.udp_allowed_sources,
            cache_minutes=settings.cache_minutes,
            max_samples=settings.udp_max_samples,
        )

    @property
    def port(self) -> int:
        return self.receiver.port

    def start(self) -> None:
        self.receiver.start()

    def stop(self) -> None:
        self.receiver.stop()

    def read(self, start: datetime, end: datetime, max_points: int) -> pd.DataFrame:
        return self.cache.window(start.timestamp(), end.timestamp(), max_points)

    def _loss(self, received: dict) -> tuple[float | None, str | None]:
        """Exact loss if the heartbeat counts telegrams, otherwise an estimate over the last minute."""
        if received["loss_percent"] is not None:
            return received["loss_percent"], "heartbeat"
        now = self._clock()
        recent = self.cache.window(now - LOSS_WINDOW_S, now, 1_000_000)
        estimate = estimate_loss_percent(epoch_seconds(recent.index))
        return estimate, None if estimate is None else "estimate"

    def stats(self) -> dict:
        """Counters for /api/health. All logged-in users (including customers) can request them: so
        no IP addresses or ports of our network. The details are in the logs and in alma.udp_check."""
        received = self.receiver.stats()
        cache = self.cache.stats()
        held = cache.newest - cache.oldest if cache.rows else 0.0
        loss, loss_source = self._loss(received)
        return {
            "listening": self.receiver.running,
            "packets": received["accepted"],
            "last_packet_age_s": received["last_packet_age_s"],
            "frozen": received["frozen"],  # True: the sender does send, but the heartbeat is stuck
            "loss_percent": None if loss is None else round(loss, 1),
            "loss_source": loss_source,
            "held_s": round(held, 1),
            "retention_s": self.retention_minutes * 60,
        }
