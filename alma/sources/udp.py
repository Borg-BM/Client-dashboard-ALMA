"""Live databron: UDP-telegrammen van iba, de laatste paar minuten in een geheugencache.

De ibaHD-server levert niet in realtime. iba (ibaPDA, Generic UDP output module) stuurt daarom elke
cyclus een telegram met alle signalen naar deze poort. De ontvanger decodeert ze volgens
udp_format.json en schrijft ze in een ringbuffer; read() leest daar het gevraagde venster uit, zodat
analyse en API niet merken dat de data live binnenkomt.

Eén proces: de ontvanger is een thread met een eigen socket. Meerdere uvicorn-workers zouden elk
dezelfde poort willen openen en de pakketten onderling verdelen; draai dit dus met één worker.
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

# 0/1-signalen: bij het inkrimpen van een venster telt het maximum, zodat een korte aan-periode blijft staan.
FLAG_SIGNALS = ("compensation_active",)

LOSS_WINDOW_S = 60  # over welke terugblik het verlies wordt geschat


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
        self.retention_minutes = cache_minutes  # de frontend biedt geen langere perioden aan
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
            # Het schip komt via Starlink (wisselend adres), dus een bronfilter is daar vaak niet bruikbaar. Dan
            # moet het telegram zelf minstens een sleutelwoord dragen, anders kan iedereen op internet meeschrijven.
            raise ValueError(
                "ALMA_UDP_ALLOW_ANY_SOURCE=true op een niet-lokaal adres vereist een 'checks'-sleutelwoord in "
                f"{settings.udp_format.name}; anders kan iedereen die de poort bereikt metingen invoeren. "
                "Gebruik liever een VPN-tunnel met ALMA_UDP_ALLOWED_SOURCES (zie de README)"
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
        """Exact verlies als de hartslag telegrammen telt, anders een schatting over de laatste minuut."""
        if received["loss_percent"] is not None:
            return received["loss_percent"], "heartbeat"
        now = self._clock()
        recent = self.cache.window(now - LOSS_WINDOW_S, now, 1_000_000)
        estimate = estimate_loss_percent(epoch_seconds(recent.index))
        return estimate, None if estimate is None else "estimate"

    def stats(self) -> dict:
        """Tellers voor /api/health. Die kunnen alle ingelogde gebruikers (ook klanten) opvragen: dus
        geen IP-adressen of poorten van ons netwerk. De details staan in de logs en in alma.udp_check."""
        received = self.receiver.stats()
        cache = self.cache.stats()
        held = cache.newest - cache.oldest if cache.rows else 0.0
        loss, loss_source = self._loss(received)
        return {
            "listening": self.receiver.running,
            "packets": received["accepted"],
            "last_packet_age_s": received["last_packet_age_s"],
            "frozen": received["frozen"],  # True: de afzender stuurt wel, maar de hartslag staat stil
            "loss_percent": None if loss is None else round(loss, 1),
            "loss_source": loss_source,
            "held_s": round(held, 1),
            "retention_s": self.retention_minutes * 60,
        }
