"""Check the connection to an ibaHD server step by step.

    python -m alma.ibahd_check                      # default paths: secrets/quick_connect.json and signals.json
    python -m alma.ibahd_check --list               # all channels, with the ID that belongs in signals.json
    python -m alma.ibahd_check --grep tower         # only channels whose name, ID or comment contains this
    python -m alma.ibahd_check --channel "store\\[0:0]" --minutes 5

In Docker: docker compose run --rm alma python -m alma.ibahd_check

Exit code 0 if everything works. The ApiKey is never printed.
"""

import argparse
import fnmatch
import ipaddress
import json
import os
import socket
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

if "--verbose" not in sys.argv:
    os.environ.setdefault("GRPC_VERBOSITY", "NONE")  # the C layer would otherwise report every failed connection separately

try:
    import grpc
    from ibahd_api import ibahd_api_pb2 as pb
except ImportError:  # pragma: no cover
    sys.exit('ibahd-api is missing. Install with: pip install -e ".[ibahd]"')

from alma.config import ROOT, SIGNALS
from alma.sources.base import SourceUnavailable
from alma.sources.ibahd import (
    CALL_TIMEOUT_S,
    connect,
    describe_rpc_error,
    fetch_channels,
    from_us,
    load_quick_connect,
    request_channel_id,
    to_us,
)

MIN_UNIX_US = 946_684_800_000_000  # 2000-01-01; anything below it, or far in the future, is not µs since 1970
STALE_AFTER = timedelta(seconds=30)
PROBE_COUNT = 3  # read this many channels from the schema while the signal map is not ready yet
MIN_RATE_HZ = 2.0  # the tower frequency is ~0.3 Hz; below this rate it is hard to determine
TYPE_NAMES = {
    pb.DATA_TYPE_FLOAT_VALUES: "float",
    pb.DATA_TYPE_DOUBLE_VALUES: "double",
    pb.DATA_TYPE_DIGITAL_VALUES: "digital",
    pb.DATA_TYPE_STRING_VALUES: "text",
}
SUPPORTED_TYPES = {pb.DATA_TYPE_FLOAT_VALUES, pb.DATA_TYPE_DOUBLE_VALUES, pb.DATA_TYPE_DIGITAL_VALUES}


class Report:
    def __init__(self, out=print, verbose: bool = False) -> None:
        self.out, self.verbose = out, verbose
        self.failures = self.warnings = self._steps = 0

    def step(self, title: str) -> None:
        self._steps += 1
        self.out(f"\n[{self._steps}] {title}")

    def ok(self, text: str) -> None:
        self.out(f"  OK     {text}")

    def info(self, text: str) -> None:
        self.out(f"         {text}")

    def warn(self, text: str) -> None:
        self.warnings += 1
        self.out(f"  WARN   {text}")

    def fail(self, text: str, hint: str | None = None) -> None:
        self.failures += 1
        self.out(f"  ERROR  {text}")
        if hint:
            self.out(f"         -> {hint}")

    def rpc_fail(self, err: grpc.RpcError, hint: str | None = None) -> None:
        self.fail(describe_rpc_error(err), hint)
        if self.verbose:
            self.info(err.debug_error_string())


@dataclass
class Channel:
    request_id: str  # as the data calls and signals.json want it
    name: str
    unit: str
    data_type: int
    active: bool
    path: str
    comments: str

    @property
    def kind(self) -> str:
        return TYPE_NAMES.get(self.data_type, pb.DataType.Name(self.data_type))


def certificate_names(pem: str) -> tuple[list[str], datetime] | None:
    """(DNS names and IPs from the certificate, expiry date), or None if it is not readable PEM."""
    from cryptography import x509

    try:
        cert = x509.load_pem_x509_certificate(pem.encode("utf-8"))
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        names = [*san.get_values_for_type(x509.DNSName), *map(str, san.get_values_for_type(x509.IPAddress))]
    except x509.ExtensionNotFound:
        names = []
    except ValueError:
        return None
    return names, cert.not_valid_after_utc


def host_in_names(host: str, names: list[str]) -> bool:
    try:
        host = str(ipaddress.ip_address(host))
    except ValueError:
        host = host.lower()
    return any(fnmatch.fnmatch(host, name.lower()) for name in names)


def check_quick_connect(r: Report, path: Path) -> tuple[dict, list[str] | None] | None:
    r.step(f"Quick-connect file ({path})")
    try:
        quick_connect = load_quick_connect(path)
    except ValueError as err:
        r.fail(str(err), "Export the quick-connect data from ibaHD Manager to this file.")
        return None
    r.ok(f"Host {quick_connect['Host']}, port {quick_connect['Port']}, ApiKey present")

    cert = certificate_names(quick_connect["TlsServerCertificate"])
    if cert is None:
        r.fail("TlsServerCertificate is not a readable PEM certificate", "Copy the whole certificate, including the BEGIN/END lines.")
        return None
    names, expires = cert
    r.info(f"Certificate valid for: {', '.join(names) or '(no names)'}; expires {expires:%Y-%m-%d}")
    if expires < datetime.now(timezone.utc):
        r.fail("The certificate has expired", "Create a new certificate and quick-connect file in ibaHD Manager.")
    if not host_in_names(str(quick_connect["Host"]), names):
        r.warn(
            f"Host {quick_connect['Host']!r} is not in the certificate; the TLS handshake will then probably "
            "fail. Use a name or IP from the list above as Host."
        )
    return quick_connect, names


def check_network(r: Report, host: str, port) -> bool:
    r.step(f"Network connection to {host}:{port}")
    try:
        with socket.create_connection((host, int(port)), timeout=5):
            pass
    except socket.gaierror:
        r.fail(f"Hostname {host!r} cannot be resolved", "Check the name (DNS or hosts file), or use the IP address.")
    except (OSError, ValueError) as err:
        r.fail(
            f"No TCP connection ({err})",
            "Is ibaHD-Server running? Is the API enabled in ibaHD Manager on this port? Does the firewall allow the port?",
        )
    else:
        r.ok("Port reachable")
        return True
    return False


def check_stores(r: Report, client, host: str, cert_names: list[str] | None) -> list[str] | None:
    r.step("Logging in and fetching HD stores (TLS + ApiKey)")
    try:
        response = client.GetHdStores(pb.GetHdStoresRequest(), timeout=CALL_TIMEOUT_S)
    except grpc.RpcError as err:
        hint = None
        if err.code() == grpc.StatusCode.UNAVAILABLE and cert_names is not None and not host_in_names(host, cert_names):
            hint = f"Probably the hostname: {host!r} is not in the certificate ({', '.join(cert_names) or 'no names'})."
        r.rpc_fail(err, hint)
        return None
    r.ok(f"Logged in; {len(response.hd_stores)} store(s)")
    time_stores = []
    for store in response.hd_stores:
        kind = pb.HdStoreType.Name(store.hd_store_type).removeprefix("HD_STORE_TYPE_").lower()
        r.info(f"{store.hd_store_name:<32} {kind:<12} enabled={store.enabled} active={store.active}")
        if store.hd_store_type == pb.HD_STORE_TYPE_TIME and store.enabled:
            time_stores.append(store.hd_store_name)
    if not time_stores:
        r.warn("No enabled time store found; so there is no measurement data to read")
    return time_stores


def collect_channels(group, store: str, path: list[str], found: list[Channel]) -> None:
    for ch in group.channels:
        found.append(
            Channel(
                request_id=request_channel_id(store, ch.channel_id),
                name=ch.name,
                unit=ch.unit,
                data_type=ch.data_type,
                active=ch.active,
                path=" / ".join(path),
                comments=f"{ch.comment_1} {ch.comment_2}".strip(),
            )
        )
    for sub in group.channel_groups:
        collect_channels(sub, store, [*path, sub.name], found)


def check_schema(r: Report, client, stores: list[str], list_all: bool, grep: str | None) -> list[Channel] | None:
    r.step("Channels from the schema of the time stores")
    channels: list[Channel] = []
    for store in stores:
        try:
            schema = client.GetHdStoreSchema(
                pb.GetHdStoreSchemaRequest(hd_store_name=store, sort_by=pb.SORT_BY_MODULE), timeout=CALL_TIMEOUT_S
            )
        except grpc.RpcError as err:
            r.rpc_fail(err, f"Cannot fetch the schema of store {store!r}; without the ibaHD-Server-API-Read licence this can fail.")
            return None
        found: list[Channel] = []
        collect_channels(schema.root_group, store, [], found)
        kinds: dict[str, int] = {}
        for ch in found:
            kinds[ch.kind] = kinds.get(ch.kind, 0) + 1
        r.ok(f"{store}: {len(found)} channels ({', '.join(f'{n} {k}' for k, n in sorted(kinds.items())) or 'empty'})")
        channels += found

    shown = channels
    if grep:
        needle = grep.lower()
        shown = [c for c in channels if needle in f"{c.request_id} {c.name} {c.path} {c.comments}".lower()]
        r.info(f"{len(shown)} of {len(channels)} channels contain {grep!r}")
    elif not list_all:
        shown = channels[:10]
    for ch in shown:
        unit = f" [{ch.unit}]" if ch.unit else ""
        flag = "" if ch.active else "  (not active)"
        r.info(f"{ch.request_id:<34} {ch.name}{unit}  {ch.kind}  {ch.path}{flag}")
    if not grep and not list_all and len(channels) > len(shown):
        r.info(f"... and {len(channels) - len(shown)} more; use --list or --grep <text>")
    r.info("signals.json takes the ID from the first column.")
    return channels


def check_signal_map(r: Report, path: Path, channels: list[Channel]) -> dict[str, Channel] | None:
    """Returns {signal: channel} if all signals are valid, otherwise None."""
    r.step(f"Signal map ({path})")
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as err:
        r.fail(f"Cannot be read: {type(err).__name__}", "See signals.json in the project folder for the format.")
        return None
    by_id = {c.request_id: c for c in channels}
    resolved: dict[str, Channel] = {}
    for signal in SIGNALS:
        value = raw.get(signal, "")
        if not value or value.startswith("<"):
            r.warn(f"{signal}: not filled in yet")
            continue
        channel = by_id.get(value)
        if channel is None:
            near = [c.request_id for c in channels if value in c.request_id or c.request_id.endswith(value)][:3]
            r.fail(
                f"{signal}: {value!r} does not occur in the schema",
                f"Did you mean: {', '.join(near)}?" if near else "Find the ID with --grep <text> or --list.",
            )
        elif channel.data_type not in SUPPORTED_TYPES:
            r.fail(f"{signal}: {value!r} has data type {channel.kind}", "Only float, double and digital channels are supported.")
        else:
            resolved[signal] = channel
            unit = f" [{channel.unit}]" if channel.unit else ""
            r.ok(f"{signal:<20} -> {value}  ({channel.name}{unit}, {channel.kind})")
            if not channel.active:
                r.warn(f"{signal}: the channel is set to not active in ibaHD")
    return resolved if len(resolved) == len(SIGNALS) else None


def channel_value_text(v) -> str:
    if v.data_type == pb.DATA_TYPE_FLOAT_VALUES:
        return f"{v.float_value:.6g}"
    if v.data_type == pb.DATA_TYPE_DOUBLE_VALUES:
        return f"{v.double_value:.6g}"
    if v.data_type == pb.DATA_TYPE_DIGITAL_VALUES:
        return f"{v.digital_value:g}"
    return pb.DataType.Name(v.data_type)


def check_last_values(r: Report, client, ids: list[str]) -> datetime | None:
    """Last value per channel. This also checks the time unit: a timestamp in the wrong unit falls
    outside the expected range. Returns the newest measurement."""
    r.step("Last value per channel")
    now = datetime.now(timezone.utc)
    try:
        response = client.GetLastRecordedChannelValue(
            pb.GetLastRecordedChannelValueRequest(time_range_from=0, time_range_to=to_us(now), channel_ids=ids),
            timeout=CALL_TIMEOUT_S,
        )
    except grpc.RpcError as err:
        r.rpc_fail(err)
        return None
    got = {v.channel_id: v for v in response.channel_values}
    newest = None
    for channel_id in ids:
        value = got.get(channel_id)
        if value is None:
            r.warn(f"{channel_id}: no value received (channel without recorded data?)")
            continue
        if not MIN_UNIX_US <= value.timestamp <= to_us(now) + 86_400_000_000:
            r.fail(
                f"{channel_id}: timestamp {value.timestamp} is not microseconds since 1970 (now is ~{to_us(now)})",
                "The server may use a different time unit; adjust to_us/from_us in alma/sources/ibahd.py.",
            )
            continue
        stamp = from_us([value.timestamp])[0].to_pydatetime()
        newest = max(newest, stamp) if newest else stamp
        r.ok(f"{channel_id:<34} {channel_value_text(value):>12}   {stamp:%Y-%m-%d %H:%M:%S}Z ({age_text(now - stamp)} ago)")
    return newest


def age_text(age: timedelta) -> str:
    seconds = age.total_seconds()
    if seconds < 120:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h" if seconds < 172_800 else f"{seconds / 86_400:.0f} days"


def check_history(r: Report, client, mapping: dict[str, str], newest: datetime | None, minutes: float) -> None:
    r.step(f"Reading a {minutes:g} min series (as the dashboard does)")
    end = datetime.now(timezone.utc)
    if newest is not None and end - newest > STALE_AFTER:
        r.warn(
            f"The newest measurement is {age_text(end - newest)} old: the dashboard would show NO DATA. "
            "I am reading the window that ends at that measurement."
        )
        end = newest + timedelta(seconds=1)
    start = end - timedelta(minutes=minutes)
    try:
        frame = fetch_channels(client, mapping, start, end, 40_000)
    except SourceUnavailable as err:
        r.fail(str(err))
        return
    if frame.empty:
        r.fail("No rows received in this window", "Is data being recorded? Try a larger window with --minutes.")
        return
    slack = timedelta(seconds=1)
    if frame.index[0] < start - slack or frame.index[-1] > end + slack:
        r.fail(f"Times {frame.index[0]} .. {frame.index[-1]} lie outside the requested window", "Points to a time unit problem.")
    median_dt = float(frame.index.to_series().diff().dt.total_seconds().median()) if len(frame) > 1 else float("nan")
    rate = 1 / median_dt if median_dt > 0 else float("nan")
    r.ok(f"{len(frame)} rows from {frame.index[0]:%H:%M:%S.%f}Z to {frame.index[-1]:%H:%M:%S.%f}Z, about {rate:.3g} Hz")
    if rate < MIN_RATE_HZ:
        r.warn(f"Below {MIN_RATE_HZ:g} Hz the tower frequency (~0.3 Hz) is hard to determine; raw data (GetRawChannelData) may be better.")
    for name in frame.columns:
        col = frame[name]
        r.info(f"{name:<22} min {col.min():.4g}  max {col.max():.4g}  mean {col.mean():.4g}")


def pick_probe_channels(channels: list[Channel]) -> dict[str, str]:
    """While the signal map is empty: a few active analog channels to test the read pipeline with."""
    analog = [c for c in channels if c.active and c.data_type in (pb.DATA_TYPE_FLOAT_VALUES, pb.DATA_TYPE_DOUBLE_VALUES)]
    return {c.name or c.request_id: c.request_id for c in analog[:PROBE_COUNT]}


def run(
    quick_connect_path: Path,
    signal_map_path: Path,
    *,
    list_all: bool = False,
    grep: str | None = None,
    channel_ids: list[str] | None = None,
    minutes: float = 1.0,
    verbose: bool = False,
    out=print,
) -> bool:
    r = Report(out, verbose)
    loaded = check_quick_connect(r, quick_connect_path)
    if loaded is None:
        return False
    quick_connect, cert_names = loaded
    if not check_network(r, str(quick_connect["Host"]), quick_connect["Port"]):
        return False

    client = connect(quick_connect)
    try:
        time_stores = check_stores(r, client, str(quick_connect["Host"]), cert_names)
        if time_stores is None:
            return False
        channels = check_schema(r, client, time_stores, list_all, grep) if time_stores else []
        if channels is None:
            return False

        resolved = check_signal_map(r, signal_map_path, channels) if channels else None
        if channel_ids:
            mapping = {cid: cid for cid in channel_ids}
        elif resolved:
            mapping = {signal: channel.request_id for signal, channel in resolved.items()}
        else:
            mapping = pick_probe_channels(channels)
            if mapping:
                r.info("Signal map not complete yet: I am testing the read with the first channels from the schema.")
        if mapping:
            newest = check_last_values(r, client, list(mapping.values()))
            check_history(r, client, mapping, newest, minutes)
    finally:
        client.close()

    r.out("\n" + "-" * 60)
    if r.failures:
        r.out(f"Result: {r.failures} error(s), {r.warnings} warning(s). Fix the ERROR lines, from top to bottom.")
        return False
    r.out(f"Result: connection works ({r.warnings} warning(s)).")
    if not resolved:
        r.out("Next step: fill signals.json with the channel IDs (see --list / --grep) and run this again.")
    else:
        r.out("Next step: set ALMA_SOURCE=ibahd and start the dashboard.")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m alma.ibahd_check", description="Check the connection to ibaHD.")
    parser.add_argument(
        "--quick-connect",
        type=Path,
        default=Path(os.environ.get("ALMA_IBAHD_QUICK_CONNECT", ROOT / "secrets" / "quick_connect.json")),
    )
    parser.add_argument(
        "--signal-map", type=Path, default=Path(os.environ.get("ALMA_SIGNAL_MAP", ROOT / "signals.json"))
    )
    parser.add_argument("--list", action="store_true", dest="list_all", help="show all channels")
    parser.add_argument("--grep", help="show only channels whose name, ID or comment contains this")
    parser.add_argument("--channel", action="append", default=[], help="read this channel ID (repeatable)")
    parser.add_argument("--minutes", type=float, default=1.0, help="length of the series to read")
    parser.add_argument("--verbose", action="store_true", help="also show the raw gRPC error message")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # a Windows console does not know every character
    ok = run(
        args.quick_connect,
        args.signal_map,
        list_all=args.list_all,
        grep=args.grep,
        channel_ids=args.channel,
        minutes=args.minutes,
        verbose=args.verbose,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
