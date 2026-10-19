"""Echte databron via de ibaHD-API (gRPC). Vereist de ibaHD-Server-API-Read licentie.

Conventies van de API (volgens het tutorial-notebook van iba): tijden zijn microseconden sinds
1970 (UTC) en kanaal-ID's hebben de vorm "<store>\\<kanaal>". Controleer dit tegen een echte server
met `python -m alma.ibahd_check`.

Aggregated data (gemiddelde per bucket) is geschikt voor trends, RMS en frequentieanalyse zolang de
buckets veel korter zijn dan een trillingsperiode: 40.000 punten over maximaal 60 minuten is ~11 Hz,
tegen een torenfrequentie van ~0,3 Hz.
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

CALL_TIMEOUT_S = 15  # zonder time-out blijft het dashboard hangen als de server niet antwoordt
QUICK_CONNECT_KEYS = ("Host", "Port", "ApiKey", "TlsServerCertificate")
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
ONE_US = timedelta(microseconds=1)

# Hoe een mislukte aanroep vermoedelijk te verklaren is. Dit zijn aannames over het gedrag van de
# server; de ruwe code en tekst van de server staan er altijd bij.
RPC_HINTS = {
    grpc.StatusCode.UNAVAILABLE: "server niet bereikbaar, of de TLS-handshake mislukt (staat de hostnaam in het certificaat?)",
    grpc.StatusCode.UNAUTHENTICATED: "ApiKey onbekend, ingetrokken of verlopen",
    grpc.StatusCode.PERMISSION_DENIED: "de ApiKey mag dit niet, of de licentie ibaHD-Server-API-Read ontbreekt of is verlopen",
    grpc.StatusCode.UNIMPLEMENTED: "de server kent deze aanroep niet (licentie of serverversie)",
    grpc.StatusCode.DEADLINE_EXCEEDED: f"de server antwoordt niet binnen {CALL_TIMEOUT_S} s",
    grpc.StatusCode.NOT_FOUND: "store of kanaal bestaat niet; controleer de namen en de kanaal-ID's",
    grpc.StatusCode.INVALID_ARGUMENT: "de server wijst de aanvraag af; controleer kanaal-ID's en tijdbereik",
}


def to_us(moment: datetime) -> int:
    """Tijdstip (met tijdzone) als microseconden sinds 1970, de eenheid van de ibaHD-API."""
    return (moment.astimezone(timezone.utc) - EPOCH) // ONE_US


def from_us(values) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(np.asarray(values, dtype=np.int64), unit="us", utc=True))


def request_channel_id(store: str, channel_id: str) -> str:
    """Kanaal-ID zoals de data-aanroepen dat willen: "<store>\\<kanaal>". Staat de store er al voor, dan
    blijft het ID ongewijzigd."""
    return channel_id if channel_id.startswith(f"{store}\\") else f"{store}\\{channel_id}"


def describe_rpc_error(err: grpc.RpcError) -> str:
    code, details = err.code(), err.details()
    hint = RPC_HINTS.get(code)
    return f"{code.name}: {details}" + (f" (meestal: {hint})" if hint else "")


def load_quick_connect(path: Path) -> dict:
    """Lees het quick-connect-bestand uit ibaHD Manager. De foutmeldingen bevatten nooit de ApiKey."""
    try:
        # utf-8-sig: Kladblok slaat JSON graag op met een BOM
        quick_connect = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise ValueError(f"Quick-connect-bestand niet gevonden: {path}") from None
    except (OSError, json.JSONDecodeError) as err:
        raise ValueError(f"Quick-connect-bestand {path} is geen geldige JSON ({type(err).__name__})") from None
    missing = [k for k in QUICK_CONNECT_KEYS if not isinstance(quick_connect, dict) or not quick_connect.get(k)]
    if missing:
        raise ValueError(f"Quick-connect-bestand {path} mist: {', '.join(missing)}")
    return quick_connect


def connect(quick_connect: dict) -> utils.HdApiClient:
    """Open een gRPC-kanaal. Dit maakt nog geen verbinding; die komt bij de eerste aanroep."""
    return utils.create_client(*utils.get_api_params(quick_connect))


def fetch_channels(
    client: utils.HdApiClient, channels: dict[str, str], start: datetime, end: datetime, max_points: int
) -> pd.DataFrame:
    """Lees de kanalen {kolomnaam: kanaal-ID} voor het venster als DataFrame met UTC-index.

    Rijen met ontbrekende waarden (gaten in de opname) vallen weg: de grafieken kunnen niet met NaN overweg.
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
        raise SourceUnavailable(f"ibaHD-server gaf geen data voor {unknown}; wel voor {sorted(by_id)}")

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
            f"Kanaal {name!r} ({channel.channel_id}) heeft datatype {pb.DataType.Name(kind)}; "
            "alleen float, double en digitaal worden ondersteund"
        )
    if len(values) != len(index):
        raise SourceUnavailable(
            f"Kanaal {name!r} ({channel.channel_id}) levert {len(values)} gemiddelden voor {len(index)} tijdstippen"
        )
    return values


def _digital_on(digital, index: pd.DatetimeIndex) -> np.ndarray:
    """Een digitaal kanaal heeft eigen tijdstippen (niet die van de analoge kanalen). Leg de toestand
    door met 'laatste bekende waarde' op het tijdraster van de rest; vóór de eerste waarde geldt de eerste."""
    if len(digital.timestamp) != len(digital.value):
        raise SourceUnavailable("Digitaal kanaal levert een ongelijk aantal tijdstippen en waarden")
    states = pd.Series(np.array(digital.value, dtype=float), index=from_us(digital.timestamp))
    states = states[~states.index.duplicated(keep="last")].sort_index()
    both = states.index.union(index)
    return states.reindex(both).ffill().bfill().reindex(index).to_numpy()


def load_signal_map(path: Path) -> dict[str, str]:
    signal_map = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    missing = [s for s in SIGNALS if s not in signal_map or signal_map[s].startswith("<")]
    if missing:
        raise ValueError(f"Signal map mist channel_id's voor: {missing}")
    return {s: signal_map[s] for s in SIGNALS}


class IbaHdSource:
    name = "ibahd"

    def __init__(self, quick_connect_path: Path, signal_map_path: Path) -> None:
        self._client = connect(load_quick_connect(quick_connect_path))
        self._map = load_signal_map(signal_map_path)

    def read(self, start: datetime, end: datetime, max_points: int) -> pd.DataFrame:
        return fetch_channels(self._client, self._map, start, end, max_points)
