from datetime import datetime
from typing import Protocol

import pandas as pd


class DataSource(Protocol):
    name: str

    def read(self, start: datetime, end: datetime, max_points: int) -> pd.DataFrame:
        """Geef data tussen start en end (UTC) terug.

        Index: UTC DatetimeIndex. Kolommen: de logische signalen uit alma.config.SIGNALS
        (accel_x/accel_y in m/s2, pressure_* in bar, system_active als 0/1).
        """
        ...
