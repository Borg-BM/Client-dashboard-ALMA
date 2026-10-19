from datetime import datetime
from typing import Protocol

import pandas as pd


class SourceUnavailable(RuntimeError):
    """The data source currently returns no usable data (server unreachable, channel missing, etc.).

    The text is meant for the logs and must not contain secrets. The API translates this into a 503.
    """


class DataSource(Protocol):
    """A source may optionally also have (the app uses them when present): retention_minutes (how far
    back the source has data, default 60), start()/stop() (app lifecycle) and stats() (counters
    for /api/health)."""

    name: str

    def read(self, start: datetime, end: datetime, max_points: int) -> pd.DataFrame:
        """Return data between start and end (UTC).

        Index: UTC DatetimeIndex. Columns: the logical signals from alma.config.SIGNALS
        (tower_*/vessel_* in mm, pressure_*_skid1/2 in bar, system_mode as mode number).
        Rows with missing values are omitted. For an unreachable source: SourceUnavailable.
        """
        ...
