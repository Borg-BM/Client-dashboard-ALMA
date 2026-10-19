"""A simulated ibaHD server for tests: real gRPC over TLS (self-signed certificate) with
API key checking, so the client runs through its real code (certificate, metadata, protobuf).

Note what this does and does not prove: the server follows the conventions as we understand them from the
iba documentation (times in microseconds since 1970, channel ID "<store>\\<channel>"). It proves that our
client is consistent with them, not that a real ibaHD server uses them that way; for that there is
`python -m alma.ibahd_check` against the real server.
"""

import datetime
import ipaddress
import time
from concurrent import futures

import grpc
import numpy as np
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from ibahd_api import ibahd_api_pb2 as pb
from ibahd_api import ibahd_api_pb2_grpc as pb_grpc

API_KEY = "test-api-key-0123456789"
STORE = "bm010"
SAMPLE_PERIOD_US = 100_000  # 10 Hz
HISTORY_S = 3600
TOGGLE_US = 5_000_000  # the digital state toggles every 5 s

FLOAT, DOUBLE, DIGITAL, STRING = (
    pb.DATA_TYPE_FLOAT_VALUES,
    pb.DATA_TYPE_DOUBLE_VALUES,
    pb.DATA_TYPE_DIGITAL_VALUES,
    pb.DATA_TYPE_STRING_VALUES,
)

# logical name -> (channel within the store, name, unit, data type)
CHANNELS = {
    "tower_x": ("[0:0]", "Tower X", "mm", FLOAT),
    "tower_y": ("[0:1]", "Tower Y", "mm", DOUBLE),
    "pressure_bore_skid1": ("[1:0]", "Pressure bore skid 1", "bar", FLOAT),
    "pressure_rod_skid1": ("[1:1]", "Pressure rod skid 1", "bar", DOUBLE),
    "pressure_bore_skid2": ("[1:2]", "Pressure bore skid 2", "bar", FLOAT),
    "pressure_rod_skid2": ("[1:3]", "Pressure rod skid 2", "bar", DOUBLE),
    "vessel_x": ("[2:0]", "Vessel X", "mm", FLOAT),
    "vessel_y": ("[2:1]", "Vessel Y", "mm", FLOAT),
    "system_mode": ("[3:0]", "System mode", "", DOUBLE),
    # Not part of SIGNALS: keeps the client's digital path under test (own timestamps per channel)
    "pump_running": ("[3:1]", "Pump running", "", DIGITAL),
}
TEXT_CHANNEL = ("[9:0]", "Operator note", "", STRING)


def full_id(name: str) -> str:
    return f"{STORE}\\{CHANNELS[name][0]}"


def analog_value(name: str, t_us) -> np.ndarray:
    """The 'recorded' value of an analog channel; deterministic, so tests can recompute it."""
    seed = sum(map(ord, name)) % 7 + 1
    return 10 * seed + 5 * np.sin(2 * np.pi * 0.3 * np.asarray(t_us) / 1e6 + seed)


def digital_state(t_us) -> np.ndarray:
    return (np.asarray(t_us) // TOGGLE_US) % 2


def make_certificate(names: tuple[str, ...]) -> tuple[bytes, bytes]:
    """Self-signed certificate for the given DNS names/IP addresses: (cert_pem, key_pem)."""
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "fake-ibahd")])
    alt_names = []
    for name in names:
        try:
            alt_names.append(x509.IPAddress(ipaddress.ip_address(name)))
        except ValueError:
            alt_names.append(x509.DNSName(name))
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    return (
        cert.public_bytes(serialization.Encoding.PEM),
        key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        ),
    )


class _ApiKeyInterceptor(grpc.ServerInterceptor):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def intercept_service(self, continuation, handler_call_details):
        sent = dict(handler_call_details.invocation_metadata).get("ibahdapi-apikey")
        if sent == self._api_key:
            return continuation(handler_call_details)

        def deny(request, context):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid API key")

        return grpc.unary_unary_rpc_method_handler(deny)


class FakeIbaHd:
    """Start a server on a free port. `quick_connect` is what ibaHD Manager would export.

    newest_age_s: how old the newest measurement is (0 = live).
    schema_ids: "full" gives channel IDs as "<store>\\[0:0]", "plain" as "[0:0]".
    cert_names: DNS names/IPs in the certificate (to simulate a hostname mismatch).
    timestamp_scale: multiplies the timestamps in 'last value' (10 = 100 ns ticks instead of µs),
        to simulate a server with a different time unit.
    """

    def __init__(
        self,
        *,
        api_key: str = API_KEY,
        newest_age_s: float = 0.0,
        schema_ids: str = "full",
        cert_names: tuple[str, ...] = ("localhost", "127.0.0.1"),
        timestamp_scale: int = 1,
    ) -> None:
        self.newest_age_s = newest_age_s
        self.schema_ids = schema_ids
        self.timestamp_scale = timestamp_scale
        self.calls: list[str] = []
        cert_pem, key_pem = make_certificate(cert_names)
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=4), interceptors=[_ApiKeyInterceptor(api_key)]
        )
        pb_grpc.add_HdApiServiceServicer_to_server(_Servicer(self), self._server)
        port = self._server.add_secure_port("127.0.0.1:0", grpc.ssl_server_credentials([(key_pem, cert_pem)]))
        self._server.start()
        self.quick_connect = {
            "Host": "localhost",
            "Port": port,
            "ApiKey": api_key,
            "TlsServerCertificate": cert_pem.decode(),
        }

    def stop(self) -> None:
        self._server.stop(grace=None).wait()

    def data_window_us(self) -> tuple[int, int]:
        """(first, last) timestamp with data, on the sample grid."""
        newest = int((time.time() - self.newest_age_s) * 1e6) // SAMPLE_PERIOD_US * SAMPLE_PERIOD_US
        return newest - HISTORY_S * 1_000_000, newest


class _Servicer(pb_grpc.HdApiServiceServicer):
    def __init__(self, fake: FakeIbaHd) -> None:
        self._fake = fake

    def _id(self, channel: str) -> str:
        return f"{STORE}\\{channel}" if self._fake.schema_ids == "full" else channel

    def _lookup(self, channel_id: str, context):
        for name, (channel, _, _, kind) in CHANNELS.items():
            if channel_id == f"{STORE}\\{channel}":
                return name, kind
        if channel_id == f"{STORE}\\{TEXT_CHANNEL[0]}":
            return "text", STRING
        context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown channel {channel_id}")

    def GetHdStores(self, request, context):
        self._fake.calls.append("GetHdStores")
        return pb.GetHdStoresResponse(
            hd_stores=[
                pb.HdStore(hd_store_name=STORE, hd_store_type=pb.HD_STORE_TYPE_TIME, enabled=True, active=True),
                pb.HdStore(hd_store_name="bm010-events", hd_store_type=pb.HD_STORE_TYPE_EVENT, enabled=True),
            ]
        )

    def GetHdStoreSchema(self, request, context):
        self._fake.calls.append("GetHdStoreSchema")
        if request.hd_store_name != STORE:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown store {request.hd_store_name}")
        modules: dict[str, list] = {}
        for channel, name, unit, kind in [*CHANNELS.values(), TEXT_CHANNEL]:
            definition = pb.ChannelDefinition(
                channel_id=self._id(channel), name=name, unit=unit, data_type=kind, active=True
            )
            modules.setdefault(channel[1], []).append(definition)
        groups = [
            pb.ChannelGroup(name=f"Module {number}", group_type=pb.GROUP_TYPE_MODULE, channels=channels)
            for number, channels in modules.items()
        ]
        root = pb.ChannelGroup(name=STORE, group_type=pb.GROUP_TYPE_HDSTORE_TIME, channel_groups=groups)
        return pb.GetHdStoreSchemaResponse(root_group=root)

    def GetAggregatedChannelData(self, request, context):
        self._fake.calls.append("GetAggregatedChannelData")
        known = [self._lookup(channel_id, context) for channel_id in request.channel_ids]
        first, last = self._fake.data_window_us()
        lo, hi = max(request.time_range_from, first), min(request.time_range_to, last)
        start = -(-lo // SAMPLE_PERIOD_US) * SAMPLE_PERIOD_US
        t = np.arange(start, hi + 1, SAMPLE_PERIOD_US, dtype=np.int64)
        if request.sample_count and len(t) > request.sample_count:
            t = t[:: -(-len(t) // request.sample_count)]

        channels = []
        for channel_id, (name, kind) in zip(request.channel_ids, known):
            out = pb.AggregatedChannel(channel_id=channel_id, data_type=kind)
            if kind == FLOAT:
                out.float_values.avg_values.extend(analog_value(name, t).tolist())
            elif kind == DOUBLE:
                out.double_values.avg_values.extend(analog_value(name, t).tolist())
            elif kind == DIGITAL:
                # own timestamps: only the toggle moments, plus the last toggle moment before the window
                edges = np.arange(lo // TOGGLE_US * TOGGLE_US, hi + 1, TOGGLE_US, dtype=np.int64)
                out.digital_values.timestamp.extend(edges.tolist())
                out.digital_values.value.extend(digital_state(edges).astype(float).tolist())
            channels.append(out)
        return pb.GetAggregatedChannelDataResponse(timestamps=t.tolist(), aggregated_channels=channels)

    def GetLastRecordedChannelValue(self, request, context):
        self._fake.calls.append("GetLastRecordedChannelValue")
        first, last = self._fake.data_window_us()
        stamp = min(request.time_range_to, last) // SAMPLE_PERIOD_US * SAMPLE_PERIOD_US
        values = []
        for channel_id in request.channel_ids:
            name, kind = self._lookup(channel_id, context)
            if stamp < first or kind == STRING:
                continue
            value = pb.ChannelValue(
                channel_id=channel_id, timestamp=int(stamp) * self._fake.timestamp_scale, data_type=kind
            )
            if kind == FLOAT:
                value.float_value = float(analog_value(name, stamp))
            elif kind == DOUBLE:
                value.double_value = float(analog_value(name, stamp))
            else:
                value.digital_value = float(digital_state(stamp))
            values.append(value)
        return pb.GetLastRecordedChannelValueResponse(channel_values=values)
