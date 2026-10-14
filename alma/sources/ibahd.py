"""Echte databron via de ibaHD-API (gRPC). Vereist de ibaHD-Server-API-Read licentie.

NOG NIET GETEST tegen een echte server; opgezet op basis van de protobuf-definities in
ibahd-api 1.3.0. Let op: aggregated data (gemiddelde per bucket) is geschikt voor trends en
RMS, maar voor nauwkeurige frequentieanalyse is GetRawChannelData beter.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from alma.config import SIGNALS

TICKS_PER_S = 10_000_000
UNIX_EPOCH_TICKS_S = 62_135_596_800  # seconden tussen 0001-01-01 en 1970-01-01


def _to_ticks(dt: datetime) -> int:
    return int((dt.timestamp() + UNIX_EPOCH_TICKS_S) * TICKS_PER_S)


def _from_ticks(ticks: np.ndarray) -> pd.DatetimeIndex:
    return pd.to_datetime(ticks / TICKS_PER_S - UNIX_EPOCH_TICKS_S, unit="s", utc=True)


class IbaHdSource:
    name = "ibahd"

    def __init__(self, quick_connect_path: Path, signal_map_path: Path) -> None:
        from ibahd_api import ibahd_api_pb2 as pb
        from ibahd_api import utils

        self._pb = pb
        qc = json.loads(Path(quick_connect_path).read_text(encoding="utf-8"))
        self._client = utils.create_client(*utils.get_api_params(qc))
        self._map: dict[str, str] = json.loads(Path(signal_map_path).read_text(encoding="utf-8"))
        missing = [s for s in SIGNALS if s not in self._map or self._map[s].startswith("<")]
        if missing:
            raise ValueError(f"Signal map mist channel_id's voor: {missing}")

    def read(self, start: datetime, end: datetime, max_points: int) -> pd.DataFrame:
        pb = self._pb
        names = list(SIGNALS)
        request = pb.GetAggregatedChannelDataRequest(
            time_range_from=_to_ticks(start),
            time_range_to=_to_ticks(end),
            channel_ids=[self._map[n] for n in names],
            sample_count=max_points,
            avg_aggregation=True,
            aggregation_algorithm_type=pb.AGGR_ALGO_TYPE_MIN_MAX_AVG_DOWNSAMPLING,
        )
        response = self._client.GetAggregatedChannelData(request)
        by_id = {c.channel_id: c for c in response.aggregated_channels}
        index = _from_ticks(np.array(response.timestamps, dtype=np.int64))
        columns = {}
        for name in names:
            ch = by_id[self._map[name]]
            values = ch.double_values if ch.double_values.avg_values else ch.float_values
            if not values.avg_values:
                raise NotImplementedError(
                    f"Kanaal {name!r} geeft geen float/double-data; digitale kanalen "
                    "(system_active) moeten nog worden uitgewerkt tegen een echte server."
                )
            columns[name] = np.array(values.avg_values)
        return pd.DataFrame(columns, index=index)
