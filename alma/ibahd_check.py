"""Controleer stap voor stap de verbinding met een ibaHD-server.

    python -m alma.ibahd_check                      # standaardpaden: secrets/quick_connect.json en signals.json
    python -m alma.ibahd_check --list               # alle kanalen, met het ID dat in signals.json moet
    python -m alma.ibahd_check --grep tower         # alleen kanalen waarvan naam, ID of commentaar dit bevat
    python -m alma.ibahd_check --channel "store\\[0:0]" --minutes 5

In Docker: docker compose run --rm alma python -m alma.ibahd_check

Exitcode 0 als alles werkt. De ApiKey wordt nooit geprint.
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
    os.environ.setdefault("GRPC_VERBOSITY", "NONE")  # de C-laag meldt anders elke mislukte verbinding apart

try:
    import grpc
    from ibahd_api import ibahd_api_pb2 as pb
except ImportError:  # pragma: no cover
    sys.exit('ibahd-api ontbreekt. Installeer met: pip install -e ".[ibahd]"')

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

MIN_UNIX_US = 946_684_800_000_000  # 2000-01-01; alles daaronder of ver in de toekomst is geen µs sinds 1970
STALE_AFTER = timedelta(seconds=30)
PROBE_COUNT = 3  # zoveel kanalen uit het schema lezen als de signaalmap nog niet klaar is
MIN_RATE_HZ = 2.0  # de torenfrequentie is ~0,3 Hz; onder dit tempo is die slecht te bepalen
TYPE_NAMES = {
    pb.DATA_TYPE_FLOAT_VALUES: "float",
    pb.DATA_TYPE_DOUBLE_VALUES: "double",
    pb.DATA_TYPE_DIGITAL_VALUES: "digitaal",
    pb.DATA_TYPE_STRING_VALUES: "tekst",
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
        self.out(f"  LET OP {text}")

    def fail(self, text: str, hint: str | None = None) -> None:
        self.failures += 1
        self.out(f"  FOUT   {text}")
        if hint:
            self.out(f"         -> {hint}")

    def rpc_fail(self, err: grpc.RpcError, hint: str | None = None) -> None:
        self.fail(describe_rpc_error(err), hint)
        if self.verbose:
            self.info(err.debug_error_string())


@dataclass
class Channel:
    request_id: str  # zoals de data-aanroepen en signals.json het willen
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
    """(DNS-namen en IP's uit het certificaat, einddatum), of None als het geen leesbaar PEM is."""
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
    r.step(f"Quick-connect-bestand ({path})")
    try:
        quick_connect = load_quick_connect(path)
    except ValueError as err:
        r.fail(str(err), "Exporteer de quick-connect-gegevens in ibaHD Manager naar dit bestand.")
        return None
    r.ok(f"Host {quick_connect['Host']}, poort {quick_connect['Port']}, ApiKey aanwezig")

    cert = certificate_names(quick_connect["TlsServerCertificate"])
    if cert is None:
        r.fail("TlsServerCertificate is geen leesbaar PEM-certificaat", "Kopieer het hele certificaat, inclusief BEGIN/END-regels.")
        return None
    names, expires = cert
    r.info(f"Certificaat geldig voor: {', '.join(names) or '(geen namen)'}; verloopt {expires:%Y-%m-%d}")
    if expires < datetime.now(timezone.utc):
        r.fail("Het certificaat is verlopen", "Maak in ibaHD Manager een nieuw certificaat en quick-connect-bestand aan.")
    if not host_in_names(str(quick_connect["Host"]), names):
        r.warn(
            f"Host {quick_connect['Host']!r} staat niet in het certificaat; de TLS-handshake mislukt dan "
            "waarschijnlijk. Gebruik als Host een naam of IP uit de lijst hierboven."
        )
    return quick_connect, names


def check_network(r: Report, host: str, port) -> bool:
    r.step(f"Netwerkverbinding met {host}:{port}")
    try:
        with socket.create_connection((host, int(port)), timeout=5):
            pass
    except socket.gaierror:
        r.fail(f"Hostnaam {host!r} is niet te vinden", "Controleer de naam (DNS of hosts-bestand), of gebruik het IP-adres.")
    except (OSError, ValueError) as err:
        r.fail(
            f"Geen TCP-verbinding ({err})",
            "Draait ibaHD-Server? Staat de API in ibaHD Manager aan op deze poort? Laat de firewall de poort door?",
        )
    else:
        r.ok("Poort bereikbaar")
        return True
    return False


def check_stores(r: Report, client, host: str, cert_names: list[str] | None) -> list[str] | None:
    r.step("Inloggen en HD-stores ophalen (TLS + ApiKey)")
    try:
        response = client.GetHdStores(pb.GetHdStoresRequest(), timeout=CALL_TIMEOUT_S)
    except grpc.RpcError as err:
        hint = None
        if err.code() == grpc.StatusCode.UNAVAILABLE and cert_names is not None and not host_in_names(host, cert_names):
            hint = f"Waarschijnlijk de hostnaam: {host!r} staat niet in het certificaat ({', '.join(cert_names) or 'geen namen'})."
        r.rpc_fail(err, hint)
        return None
    r.ok(f"Ingelogd; {len(response.hd_stores)} store(s)")
    time_stores = []
    for store in response.hd_stores:
        kind = pb.HdStoreType.Name(store.hd_store_type).removeprefix("HD_STORE_TYPE_").lower()
        r.info(f"{store.hd_store_name:<32} {kind:<12} ingeschakeld={store.enabled} actief={store.active}")
        if store.hd_store_type == pb.HD_STORE_TYPE_TIME and store.enabled:
            time_stores.append(store.hd_store_name)
    if not time_stores:
        r.warn("Geen ingeschakelde tijd-store gevonden; er is dus geen meetdata om te lezen")
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
    r.step("Kanalen uit het schema van de tijd-stores")
    channels: list[Channel] = []
    for store in stores:
        try:
            schema = client.GetHdStoreSchema(
                pb.GetHdStoreSchemaRequest(hd_store_name=store, sort_by=pb.SORT_BY_MODULE), timeout=CALL_TIMEOUT_S
            )
        except grpc.RpcError as err:
            r.rpc_fail(err, f"Schema van store {store!r} niet op te halen; zonder licentie ibaHD-Server-API-Read kan dat mislukken.")
            return None
        found: list[Channel] = []
        collect_channels(schema.root_group, store, [], found)
        kinds: dict[str, int] = {}
        for ch in found:
            kinds[ch.kind] = kinds.get(ch.kind, 0) + 1
        r.ok(f"{store}: {len(found)} kanalen ({', '.join(f'{n} {k}' for k, n in sorted(kinds.items())) or 'leeg'})")
        channels += found

    shown = channels
    if grep:
        needle = grep.lower()
        shown = [c for c in channels if needle in f"{c.request_id} {c.name} {c.path} {c.comments}".lower()]
        r.info(f"{len(shown)} van {len(channels)} kanalen bevatten {grep!r}")
    elif not list_all:
        shown = channels[:10]
    for ch in shown:
        unit = f" [{ch.unit}]" if ch.unit else ""
        flag = "" if ch.active else "  (niet actief)"
        r.info(f"{ch.request_id:<34} {ch.name}{unit}  {ch.kind}  {ch.path}{flag}")
    if not grep and not list_all and len(channels) > len(shown):
        r.info(f"... en {len(channels) - len(shown)} meer; gebruik --list of --grep <tekst>")
    r.info("In signals.json hoort het ID uit de eerste kolom.")
    return channels


def check_signal_map(r: Report, path: Path, channels: list[Channel]) -> dict[str, Channel] | None:
    """Geeft {signaal: kanaal} als alle signalen kloppen, anders None."""
    r.step(f"Signaalmap ({path})")
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as err:
        r.fail(f"Niet te lezen: {type(err).__name__}", "Zie signals.json in de projectmap voor het formaat.")
        return None
    by_id = {c.request_id: c for c in channels}
    resolved: dict[str, Channel] = {}
    for signal in SIGNALS:
        value = raw.get(signal, "")
        if not value or value.startswith("<"):
            r.warn(f"{signal}: nog niet ingevuld")
            continue
        channel = by_id.get(value)
        if channel is None:
            near = [c.request_id for c in channels if value in c.request_id or c.request_id.endswith(value)][:3]
            r.fail(
                f"{signal}: {value!r} komt niet voor in het schema",
                f"Bedoelde je: {', '.join(near)}?" if near else "Zoek het ID met --grep <tekst> of --list.",
            )
        elif channel.data_type not in SUPPORTED_TYPES:
            r.fail(f"{signal}: {value!r} heeft datatype {channel.kind}", "Alleen float-, double- en digitale kanalen worden ondersteund.")
        else:
            resolved[signal] = channel
            unit = f" [{channel.unit}]" if channel.unit else ""
            r.ok(f"{signal:<20} -> {value}  ({channel.name}{unit}, {channel.kind})")
            if not channel.active:
                r.warn(f"{signal}: het kanaal staat in ibaHD op niet actief")
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
    """Laatste waarde per kanaal. Dit controleert ook de tijdeenheid: een tijdstempel in de verkeerde
    eenheid valt buiten het verwachte bereik. Geeft de nieuwste meting terug."""
    r.step("Laatste waarde per kanaal")
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
            r.warn(f"{channel_id}: geen waarde ontvangen (kanaal zonder opgenomen data?)")
            continue
        if not MIN_UNIX_US <= value.timestamp <= to_us(now) + 86_400_000_000:
            r.fail(
                f"{channel_id}: tijdstempel {value.timestamp} is geen microseconden sinds 1970 (nu is ~{to_us(now)})",
                "De server gebruikt mogelijk een andere tijdeenheid; pas to_us/from_us in alma/sources/ibahd.py aan.",
            )
            continue
        stamp = from_us([value.timestamp])[0].to_pydatetime()
        newest = max(newest, stamp) if newest else stamp
        r.ok(f"{channel_id:<34} {channel_value_text(value):>12}   {stamp:%Y-%m-%d %H:%M:%S}Z ({age_text(now - stamp)} geleden)")
    return newest


def age_text(age: timedelta) -> str:
    seconds = age.total_seconds()
    if seconds < 120:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} uur" if seconds < 172_800 else f"{seconds / 86_400:.0f} dagen"


def check_history(r: Report, client, mapping: dict[str, str], newest: datetime | None, minutes: float) -> None:
    r.step(f"Meetreeks van {minutes:g} min uitlezen (zoals het dashboard dat doet)")
    end = datetime.now(timezone.utc)
    if newest is not None and end - newest > STALE_AFTER:
        r.warn(
            f"De nieuwste meting is {age_text(end - newest)} oud: het dashboard zou NO DATA tonen. "
            "Ik lees het venster dat op die meting eindigt."
        )
        end = newest + timedelta(seconds=1)
    start = end - timedelta(minutes=minutes)
    try:
        frame = fetch_channels(client, mapping, start, end, 40_000)
    except SourceUnavailable as err:
        r.fail(str(err))
        return
    if frame.empty:
        r.fail("Geen rijen ontvangen in dit venster", "Wordt er data opgenomen? Probeer een groter venster met --minutes.")
        return
    slack = timedelta(seconds=1)
    if frame.index[0] < start - slack or frame.index[-1] > end + slack:
        r.fail(f"Tijden {frame.index[0]} .. {frame.index[-1]} liggen buiten het gevraagde venster", "Wijst op een tijdeenheid-probleem.")
    median_dt = float(frame.index.to_series().diff().dt.total_seconds().median()) if len(frame) > 1 else float("nan")
    rate = 1 / median_dt if median_dt > 0 else float("nan")
    r.ok(f"{len(frame)} rijen van {frame.index[0]:%H:%M:%S.%f}Z tot {frame.index[-1]:%H:%M:%S.%f}Z, ongeveer {rate:.3g} Hz")
    if rate < MIN_RATE_HZ:
        r.warn(f"Onder {MIN_RATE_HZ:g} Hz is de torenfrequentie (~0,3 Hz) slecht te bepalen; ruwe data (GetRawChannelData) kan beter zijn.")
    for name in frame.columns:
        col = frame[name]
        r.info(f"{name:<22} min {col.min():.4g}  max {col.max():.4g}  gemiddeld {col.mean():.4g}")


def pick_probe_channels(channels: list[Channel]) -> dict[str, str]:
    """Zolang de signaalmap leeg is: een paar actieve analoge kanalen om de leespijplijn mee te testen."""
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
                r.info("Signaalmap nog niet compleet: ik test het uitlezen met de eerste kanalen uit het schema.")
        if mapping:
            newest = check_last_values(r, client, list(mapping.values()))
            check_history(r, client, mapping, newest, minutes)
    finally:
        client.close()

    r.out("\n" + "-" * 60)
    if r.failures:
        r.out(f"Resultaat: {r.failures} fout(en), {r.warnings} waarschuwing(en). Los de FOUT-regels op, van boven naar beneden.")
        return False
    r.out(f"Resultaat: verbinding werkt ({r.warnings} waarschuwing(en)).")
    if not resolved:
        r.out("Volgende stap: vul signals.json met de kanaal-ID's (zie --list / --grep) en draai dit opnieuw.")
    else:
        r.out("Volgende stap: zet ALMA_SOURCE=ibahd en start het dashboard.")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m alma.ibahd_check", description="Controleer de verbinding met ibaHD.")
    parser.add_argument(
        "--quick-connect",
        type=Path,
        default=Path(os.environ.get("ALMA_IBAHD_QUICK_CONNECT", ROOT / "secrets" / "quick_connect.json")),
    )
    parser.add_argument(
        "--signal-map", type=Path, default=Path(os.environ.get("ALMA_SIGNAL_MAP", ROOT / "signals.json"))
    )
    parser.add_argument("--list", action="store_true", dest="list_all", help="toon alle kanalen")
    parser.add_argument("--grep", help="toon alleen kanalen waarvan naam, ID of commentaar dit bevat")
    parser.add_argument("--channel", action="append", default=[], help="lees dit kanaal-ID uit (herhaalbaar)")
    parser.add_argument("--minutes", type=float, default=1.0, help="lengte van de uitgelezen meetreeks")
    parser.add_argument("--verbose", action="store_true", help="toon ook de ruwe gRPC-foutmelding")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # een Windows-console kent niet elk teken
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
