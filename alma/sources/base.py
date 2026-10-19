from datetime import datetime
from typing import Protocol

import pandas as pd


class SourceUnavailable(RuntimeError):
    """De databron geeft nu geen bruikbare data (server onbereikbaar, kanaal ontbreekt, enz.).

    De tekst is bedoeld voor de logs en mag geen geheimen bevatten. De API vertaalt dit naar een 503.
    """


class DataSource(Protocol):
    """Een bron kan daarnaast optioneel hebben (de app gebruikt ze als ze er zijn): retention_minutes (hoe ver
    terug de bron data heeft, standaard 60), start()/stop() (levenscyclus van de app) en stats() (tellers
    voor /api/health)."""

    name: str

    def read(self, start: datetime, end: datetime, max_points: int) -> pd.DataFrame:
        """Geef data tussen start en end (UTC) terug.

        Index: UTC DatetimeIndex. Kolommen: de logische signalen uit alma.config.SIGNALS
        (tower_*/vessel_* in mm, pressure_*_skid1/2 in bar, system_mode als modenummer).
        Rijen met ontbrekende waarden worden weggelaten. Bij een onbereikbare bron: SourceUnavailable.
        """
        ...
