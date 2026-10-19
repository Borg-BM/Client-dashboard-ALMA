"""Controleer de eerste verbinding met iba: luister een paar seconden op de UDP-poort en laat zien wie er
stuurt, hoe vaak en hoe regelmatig, of de telegrammen bij het formaat passen en welke waarden erin zitten.

    python -m alma.udp_check                  luister 10 s op de poort uit de instellingen
    python -m alma.udp_check --dump           toon ook de ruwe bytes van het eerste telegram
    python -m alma.udp_check --port 5006 --seconds 30

De tool opent de poort zelf: stop het dashboard eerst, of kies een andere poort met --port. Hij filtert
niet op bronadres; het rapport noemt de adressen die je in ALMA_UDP_ALLOWED_SOURCES moet zetten.

Afsluitcode: 0 = telegrammen ontvangen en ze passen; 1 = ze komen binnen maar passen niet (of het formaat is
ongeldig); 2 = niets ontvangen of de poort kon niet worden geopend.
"""

import argparse
import ipaddress
import math
import socket
import statistics
import struct
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from alma.config import ROOT, SIGNALS, Settings
from alma.ingest.format import MAX_PACKET_BYTES, FormatError, PacketError, PacketFormat

MAX_DUMP_ROWS = 128


@dataclass
class Sender:
    """Alles wat we van een bronadres zien."""

    packets: int = 0
    ports: set[int] = field(default_factory=set)
    sizes: Counter = field(default_factory=Counter)
    arrivals: list[float] = field(default_factory=list)  # time.monotonic()
    fits: int = 0
    errors: Counter = field(default_factory=Counter)
    first: bytes | None = None
    low: dict[str, float] = field(default_factory=dict)
    high: dict[str, float] = field(default_factory=dict)
    total: dict[str, float] = field(default_factory=dict)
    last: dict[str, float] = field(default_factory=dict)

    def add(self, address: tuple, data: bytes, arrived: float, fmt: PacketFormat | None) -> None:
        self.packets += 1
        self.ports.add(address[1])
        self.sizes[len(data)] += 1
        self.arrivals.append(arrived)
        if self.first is None:
            self.first = data
        if fmt is None:
            return
        try:
            values = fmt.decode(data)
        except PacketError as err:
            self.errors[str(err)] += 1
            return
        self.fits += 1
        for name, value in values.items():
            self.low[name] = min(self.low.get(name, value), value)
            self.high[name] = max(self.high.get(name, value), value)
            self.total[name] = self.total.get(name, 0.0) + value
            self.last[name] = value


def _defaults() -> tuple[str, int, Path]:
    """Poort, adres en formaatbestand uit de instellingen (.env en omgeving), zonder dat een andere
    ALMA_SOURCE of een ontbrekende login de tool in de weg zit."""
    try:
        settings = Settings(source="mock", auth_enabled=False)
    except ValueError:
        return "0.0.0.0", 5005, ROOT / "udp_format.json"
    return settings.udp_bind, settings.udp_port, settings.udp_format


def listen(sock: socket.socket, seconds: float, fmt: PacketFormat | None) -> tuple[dict[str, Sender], float]:
    """Luister `seconds` lang (of tot Ctrl+C); geef de afzenders per IP en de werkelijke duur."""
    senders: dict[str, Sender] = {}
    buffer = bytearray(MAX_PACKET_BYTES + 1)
    started = time.monotonic()
    deadline = started + seconds
    try:
        while (left := deadline - time.monotonic()) > 0:
            sock.settimeout(min(left, 0.5))
            try:
                size, address = sock.recvfrom_into(buffer)
            except socket.timeout:
                continue
            arrived = time.monotonic()
            senders.setdefault(address[0], Sender()).add(address, bytes(buffer[:size]), arrived, fmt)
    except KeyboardInterrupt:
        print("\nOnderbroken; dit is het rapport tot nu toe.")
    return senders, time.monotonic() - started


def _rhythm(sender: Sender) -> str:
    if sender.packets < 2:
        return "te weinig telegrammen voor een ritme"
    gaps = [b - a for a, b in zip(sender.arrivals, sender.arrivals[1:])]
    span = sender.arrivals[-1] - sender.arrivals[0]
    rate = (sender.packets - 1) / span if span > 0 else math.inf
    return (
        f"{rate:.1f} per seconde; interval mediaan {statistics.median(gaps) * 1000:.0f} ms, "
        f"langste gat {max(gaps) * 1000:.0f} ms"
    )


def _print_sender(host: str, sender: Sender, fmt: PacketFormat | None) -> None:
    ports = ", ".join(str(p) for p in sorted(sender.ports))
    print(f"Bron {host} (poort {ports})")
    print(f"  telegrammen   : {sender.packets}; {_rhythm(sender)}")
    sizes = ", ".join(f"{size} bytes ({n}x)" for size, n in sorted(sender.sizes.items()))
    wanted = "" if fmt is None else f"  [formaat verwacht {fmt.packet_bytes} bytes]"
    print(f"  lengte        : {sizes}{wanted}")
    if fmt is None:
        return
    print(f"  passen bij het formaat: {sender.fits} van {sender.packets}")
    for reason, n in sender.errors.most_common(3):
        print(f"    {n}x geweigerd: {reason}")
    if sender.fits:
        print(f"  {'signaal':<22}{'min':>12}{'gemiddeld':>12}{'max':>12}{'laatst':>12}")
        for name in fmt.signals:
            mean = sender.total[name] / sender.fits
            print(
                f"  {name:<22}{sender.low[name]:>12.4g}{mean:>12.4g}{sender.high[name]:>12.4g}{sender.last[name]:>12.4g}"
            )
    print()


def dump(packet: bytes, step: int = 4) -> str:
    """De bytes van een telegram als tabel: per offset de vier bytes als hex en als float32 en int32, in
    beide byte-volgordes. Met het oog is hier een verkeerde offset of byte-volgorde in te vinden."""
    lines = [
        f"Telegram van {len(packet)} bytes: {packet.hex(' ')}",
        f"{'offset':>6}  {'bytes':<11}{'FLOAT big':>14}{'FLOAT little':>14}{'DINT big':>13}{'DINT little':>13}",
    ]
    offsets = range(0, len(packet) - 3, step)
    for offset in offsets[:MAX_DUMP_ROWS]:
        chunk = packet[offset : offset + 4]
        big_f, little_f = struct.unpack(">f", chunk)[0], struct.unpack("<f", chunk)[0]
        big_i, little_i = struct.unpack(">i", chunk)[0], struct.unpack("<i", chunk)[0]
        lines.append(
            f"{offset:>6}  {chunk.hex(' '):<11}{big_f:>14.5g}{little_f:>14.5g}{big_i:>13d}{little_i:>13d}"
        )
    if len(offsets) > MAX_DUMP_ROWS:
        lines.append(f"... alleen de eerste {MAX_DUMP_ROWS} regels; kies een grotere --step voor de rest")
    return "\n".join(lines)


def _open(bind: str, port: int) -> socket.socket:
    family = socket.AF_INET6 if ipaddress.ip_address(bind).version == 6 else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_DGRAM)
    try:
        sock.bind((bind, port))
    except OSError:
        sock.close()
        raise
    return sock


def main(argv: list[str] | None = None) -> int:
    default_bind, default_port, default_format = _defaults()
    parser = argparse.ArgumentParser(
        prog="python -m alma.udp_check", description="Luister op de UDP-poort en controleer de telegrammen van iba."
    )
    parser.add_argument("--port", type=int, default=default_port, help=f"UDP-poort (standaard {default_port})")
    parser.add_argument("--bind", default=default_bind, help=f"luisteradres (standaard {default_bind})")
    parser.add_argument("--seconds", type=float, default=10.0, help="hoe lang er wordt geluisterd (standaard 10)")
    parser.add_argument("--format", type=Path, default=default_format, help="telegramformaat (udp_format.json)")
    parser.add_argument("--dump", action="store_true", help="toon de ruwe bytes van het eerste telegram")
    parser.add_argument("--step", type=int, choices=(1, 2, 4), default=4, help="afstand tussen de offsets bij --dump")
    args = parser.parse_args(argv)

    try:
        ipaddress.ip_address(args.bind)
    except ValueError:
        parser.error("--bind moet een IP-adres zijn, bv. 0.0.0.0")

    fmt: PacketFormat | None
    try:
        fmt = PacketFormat.load(args.format, required_signals=SIGNALS)
    except FormatError as err:
        print(f"Het formaat is niet bruikbaar: {err}\nDe tool laat alleen zien wat er binnenkomt.\n")
        fmt = None

    try:
        sock = _open(args.bind, args.port)
    except OSError as err:
        print(
            f"Kan UDP-poort {args.port} op {args.bind} niet openen ({err.strerror or err}). Stop het dashboard "
            "(of een andere ontvanger) of kies een andere poort met --port."
        )
        return 2

    print(f"Luisteren op {args.bind}:{sock.getsockname()[1]} gedurende {args.seconds:g} s (Ctrl+C om te stoppen)...\n")
    try:
        senders, elapsed = listen(sock, args.seconds, fmt)
    finally:
        sock.close()

    if not senders:
        print(f"Niets ontvangen in {elapsed:.0f} s. Controleer:")
        print(f"  - dat ibaPDA naar het IP-adres van deze machine stuurt, poort {args.port} (bij een VPN: het tunnel-IP);")
        print("  - dat de firewall (Windows: inkomend UDP voor python.exe) deze poort toelaat;")
        print("  - dat er geen ander programma op deze poort luistert en dat de uitvoermodule in ibaPDA actief is.")
        return 2

    for host, sender in sorted(senders.items(), key=lambda item: -item[1].packets):
        _print_sender(host, sender, fmt)
    if args.dump:
        first = next(s.first for s in senders.values() if s.first is not None)
        print(dump(first, args.step) + "\n")

    hosts = ",".join(sorted(senders))
    if fmt is None:
        return 1
    if all(s.fits == s.packets for s in senders.values()):
        print("Alles past bij het formaat.")
        print(f"Voor het dashboard: ALMA_UDP_ALLOWED_SOURCES={hosts}")
        return 0
    print("Niet alle telegrammen passen bij het formaat. Controleer lengte, offsets en byte-volgorde in ibaPDA")
    print("en in udp_format.json; met --dump zie je de ruwe bytes in beide byte-volgordes.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
