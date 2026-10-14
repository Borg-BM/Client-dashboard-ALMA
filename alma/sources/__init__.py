from alma.config import Settings
from alma.sources.base import DataSource


def create_source(settings: Settings) -> DataSource:
    if settings.source == "mock":
        from alma.sources.mock import MockSource

        return MockSource()
    if settings.source == "ibahd":
        from alma.sources.ibahd import IbaHdSource

        return IbaHdSource(settings.ibahd_quick_connect, settings.signal_map)
    raise ValueError(f"Onbekende ALMA_SOURCE: {settings.source!r} (kies 'mock' of 'ibahd')")
