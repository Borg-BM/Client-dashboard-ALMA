"""Real data source via the ibaHD API (gRPC). Requires the ibaHD-Server-API-Read licence.

API conventions (per iba's tutorial notebook): times are microseconds since 1970 (UTC) and
channel IDs have the form "<store>\\<channel>". Verify this against a real server with
`python -m alma.ibahd_check`.

Aggregated data (average per bucket) is suitable for trends, RMS and frequency analysis as long as the
buckets are much shorter than a vibration period: 40,000 points over at most 60 minutes is ~11 Hz,
against a tower frequency of ~0.3 Hz.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import grpc
import numpy as np
import pandas as pd
from ibahd_api import ibahd_api_pb2 as pb
from ibahd_api import utils

from alma.config import SIGNALS
from alma.sources.base import SourceUnavailable

CALL_TIMEOUT_S = 15  # without a timeout the dashboard hangs when the server does not answer
QUICK_CONNECT_KEYS = ("Host", "Port", "ApiKey", "TlsServerCertificate")
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
ONE_US = timedelta(microseconds=1)

# How a failed call can probably be explained. These are assumptions about the behaviour of the
# server; the raw code and text from the server are always included.
RPC_HINTS = {
    grpc.StatusCode.UNAVAILABLE: "server unreachable, or the TLS handshake failed (is the hostname in the certificate?)",
    grpc.StatusCode.UNAUTHENTICATED: "ApiKey unknown, revoked or expired",
    grpc.StatusCode.PERMISSION_DENIED: "the ApiKey is not allowed to do this, or the ibaHD-Server-API-Read licence is missing or has expired",
    grpc.StatusCode.UNIMPLEMENTED: "the server does not know this call (licence or server version)",
    grpc.StatusCode.DEADLINE_EXCEEDED: f"the server does not answer within {CALL_TIMEOUT_S} s",
    grpc.StatusCode.NOT_FOUND: "store or channel does not exist; check the names and the channel IDs",
    grpc.StatusCode.INVALID_ARGUMENT: "the server rejects the request; check channel IDs and time range",
}


def to_us(moment: datetime) -> int:
    """Moment in time (timezone-aware) as microseconds since 1970, the unit of the ibaHD API."""
    return (moment.astimezone(timezone.utc) - EPOCH) // ONE_US


def from_us(values) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(np.asarray(values, dtype=np.int64), unit="us", utc=True))


def request_channel_id(store: str, channel_id: str) -> str:
    """Channel ID as the data calls want it: "<store>\\<channel>". If the store is already in front,
    the ID is left unchanged."""
    return channel_id if channel_id.startswith(f"{store}\\") else f"{store}\\{channel_id}"


def describe_rpc_error(err: grpc.RpcError) -> str:
    code, details = err.code(), err.details()
    hint = RPC_HINTS.get(code)
    return f"{code.name}: {details}" + (f" (usually: {hint})" if hint else "")


def load_quick_connect(path: Path) -> dict:
    """Read the quick-connect file from ibaHD Manager. The error messages never contain the ApiKey."""
    try:
        # utf-8-sig: Notepad likes to save JSON with a BOM
        quick_connect = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise ValueError(f"Quick-connect file not found: {path}") from None
    except (OSError, json.JSONDecodeError) as err:
        raise ValueError(f"Quick-connect file {path} is not valid JSON ({type(err).__name__})") from None
    missing = [k for k in QUICK_CONNECT_KEYS if not isinstance(quick_connect, dict) or not quick_connect.get(k)]
    if missing:
        raise ValueError(f"Quick-connect file {path} is missing: {', '.join(missing)}")
    return quick_connect


def connect(quick_connect: dict) -> utils.HdApiClient:
    """Open a gRPC channel. This does not connect yet; that happens on the first call."""
    return utils.create_client(*utils.get_api_params(quick_connect))


def fetch_channels(
    client: utils.HdApiClient, channels: dict[str, str], start: datetime, end: datetime, max_points: int
) -> pd.DataFrame:
    """Read the channels {column name: channel ID} for the window as a DataFrame with a UTC index.

    Rows with missing values (gaps in the recording) are dropped: the charts cannot handle NaN.
    """
    names = list(channels)
    request = pb.GetAggregatedChannelDataRequest(
        time_range_from=to_us(start),
        time_range_to=to_us(end),
        channel_ids=[channels[n] for n in names],
        sample_count=max_points,
        avg_aggregation=True,
        aggregation_algorithm_type=pb.AGGR_ALGO_TYPE_MIN_MAX_AVG_DOWNSAMPLING,
    )
    try:
        response = client.GetAggregatedChannelData(request, timeout=CALL_TIMEOUT_S)
    except grpc.RpcError as err:
        raise SourceUnavailable(f"ibaHD-server: {describe_rpc_error(err)}") from err

    by_id = {c.channel_id: c for c in response.aggregated_channels}
    unknown = [channels[n] for n in names if channels[n] not in by_id]
    if unknown:
        raise SourceUnavailable(f"ibaHD-server returned no data for {unknown}; it did for {sorted(by_id)}")

    index = from_us(response.timestamps)
    columns = {name: _column(by_id[channels[name]], name, index) for name in names}
    frame = pd.DataFrame(columns, index=index)
    return frame.dropna()


def _column(channel, name: str, index: pd.DatetimeIndex) -> np.ndarray:
    kind = channel.data_type
    if kind == pb.DATA_TYPE_FLOAT_VALUES:
        values = np.array(channel.float_values.avg_values, dtype=float)
    elif kind == pb.DATA_TYPE_DOUBLE_VALUES:
        values = np.array(channel.double_values.avg_values, dtype=float)
    elif kind == pb.DATA_TYPE_DIGITAL_VALUES:
        return _digital_on(channel.digital_values, index)
    else:
        raise SourceUnavailable(
            f"Channel {name!r} ({channel.channel_id}) has data type {pb.DataType.Name(kind)}; "
            "only float, double and digital are supported"
        )
    if len(values) != len(index):
        raise SourceUnavailable(
            f"Channel {name!r} ({channel.channel_id}) returns {len(values)} averages for {len(index)} timestamps"
        )
    return values


def _digital_on(digital, index: pd.DatetimeIndex) -> np.ndarray:
    """A digital channel has its own timestamps (not those of the analog channels). Carry the state
    forward ('last known value') onto the time grid of the rest; before the first value, the first one applies."""
    if len(digital.timestamp) != len(digital.value):
        raise SourceUnavailable("Digital channel returns an unequal number of timestamps and values")
    states = pd.Series(np.array(digital.value, dtype=float), index=from_us(digital.timestamp))
    states = states[~states.index.duplicated(keep="last")].sort_index()
    both = states.index.union(index)
    return states.reindex(both).ffill().bfill().reindex(index).to_numpy()


def load_signal_map(path: Path) -> dict[str, str]:
    signal_map = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    missing = [s for s in SIGNALS if s not in signal_map or signal_map[s].startswith("<")]
    if missing:
        raise ValueError(f"Signal map is missing channel IDs for: {missing}")
    return {s: signal_map[s] for s in SIGNALS}


class IbaHdSource:
    name = "ibahd"

    def __init__(self, quick_connect_path: Path, signal_map_path: Path) -> None:
        self._client = connect(load_quick_connect(quick_connect_path))
        self._map = load_signal_map(signal_map_path)

    def read(self, start: datetime, end: datetime, max_points: int) -> pd.DataFrame:
        return fetch_channels(self._client, self._map, start, end, max_points)
