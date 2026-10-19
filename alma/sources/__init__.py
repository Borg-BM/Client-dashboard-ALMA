from alma.config import Settings
from alma.sources.base import DataSource


def create_source(settings: Settings) -> DataSource:
    if settings.source == "mock":
        from alma.sources.mock import MockSource

        return MockSource()
    if settings.source == "ibahd":
        from alma.sources.ibahd import IbaHdSource

        return IbaHdSource(settings.ibahd_quick_connect, settings.signal_map)
    if settings.source == "udp":
        from alma.sources.udp import UdpSource

        return UdpSource.from_settings(settings)
    raise ValueError(f"Onbekende ALMA_SOURCE: {settings.source!r} (kies 'mock', 'ibahd' of 'udp')")
