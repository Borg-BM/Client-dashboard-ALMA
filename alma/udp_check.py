"""Check the first connection with iba: listen on the UDP port for a few seconds and show who is sending,
how often and how regularly, whether the telegrams fit the format and which values they contain.

    python -m alma.udp_check                  listen for 10 s on the port from the settings
    python -m alma.udp_check --dump           also show the raw bytes of the first telegram
    python -m alma.udp_check --port 5006 --seconds 30

The tool opens the port itself: stop the dashboard first, or choose another port with --port. It does not
filter on source address; the report names the addresses you need to put in ALMA_UDP_ALLOWED_SOURCES.

Exit code: 0 = telegrams received and they fit; 1 = they arrive but do not fit (or the format is
invalid); 2 = nothing received or the port could not be opened.
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
    """Everything we see from one source address."""

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
    """Port, address and format file from the settings (.env and environment), without another
    ALMA_SOURCE or a missing login getting in the tool's way."""
    try:
        settings = Settings(source="mock", auth_enabled=False)
    except ValueError:
        return "0.0.0.0", 5005, ROOT / "udp_format.json"
    return settings.udp_bind, settings.udp_port, settings.udp_format


def listen(sock: socket.socket, seconds: float, fmt: PacketFormat | None) -> tuple[dict[str, Sender], float]:
    """Listen for `seconds` (or until Ctrl+C); return the senders per IP and the actual duration."""
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
        print("\nInterrupted; this is the report so far.")
    return senders, time.monotonic() - started


def _rhythm(sender: Sender) -> str:
    if sender.packets < 2:
        return "too few telegrams for a rhythm"
    gaps = [b - a for a, b in zip(sender.arrivals, sender.arrivals[1:])]
    span = sender.arrivals[-1] - sender.arrivals[0]
    rate = (sender.packets - 1) / span if span > 0 else math.inf
    return (
        f"{rate:.1f} per second; interval median {statistics.median(gaps) * 1000:.0f} ms, "
        f"longest gap {max(gaps) * 1000:.0f} ms"
    )


def _print_sender(host: str, sender: Sender, fmt: PacketFormat | None) -> None:
    ports = ", ".join(str(p) for p in sorted(sender.ports))
    print(f"Source {host} (port {ports})")
    print(f"  telegrams     : {sender.packets}; {_rhythm(sender)}")
    sizes = ", ".join(f"{size} bytes ({n}x)" for size, n in sorted(sender.sizes.items()))
    wanted = "" if fmt is None else f"  [format expects {fmt.packet_bytes} bytes]"
    print(f"  length        : {sizes}{wanted}")
    if fmt is None:
        return
    print(f"  fit the format: {sender.fits} of {sender.packets}")
    for reason, n in sender.errors.most_common(3):
        print(f"    {n}x refused: {reason}")
    if sender.fits:
        print(f"  {'signal':<22}{'min':>12}{'mean':>12}{'max':>12}{'last':>12}")
        for name in fmt.signals:
            mean = sender.total[name] / sender.fits
            print(
                f"  {name:<22}{sender.low[name]:>12.4g}{mean:>12.4g}{sender.high[name]:>12.4g}{sender.last[name]:>12.4g}"
            )
    print()


def dump(packet: bytes, step: int = 4) -> str:
    """The bytes of a telegram as a table: per offset the four bytes as hex and as float32 and int32, in
    both byte orders. By eye, a wrong offset or byte order can be spotted here."""
    lines = [
        f"Telegram of {len(packet)} bytes: {packet.hex(' ')}",
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
        lines.append(f"... only the first {MAX_DUMP_ROWS} rows; choose a larger --step for the rest")
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
        prog="python -m alma.udp_check", description="Listen on the UDP port and check the telegrams from iba."
    )
    parser.add_argument("--port", type=int, default=default_port, help=f"UDP port (default {default_port})")
    parser.add_argument("--bind", default=default_bind, help=f"listen address (default {default_bind})")
    parser.add_argument("--seconds", type=float, default=10.0, help="how long to listen (default 10)")
    parser.add_argument("--format", type=Path, default=default_format, help="telegram format (udp_format.json)")
    parser.add_argument("--dump", action="store_true", help="show the raw bytes of the first telegram")
    parser.add_argument("--step", type=int, choices=(1, 2, 4), default=4, help="offset spacing for --dump")
    args = parser.parse_args(argv)

    try:
        ipaddress.ip_address(args.bind)
    except ValueError:
        parser.error("--bind must be an IP address, e.g. 0.0.0.0")

    fmt: PacketFormat | None
    try:
        fmt = PacketFormat.load(args.format, required_signals=SIGNALS)
    except FormatError as err:
        print(f"The format is not usable: {err}\nThe tool will only show what arrives.\n")
        fmt = None

    try:
        sock = _open(args.bind, args.port)
    except OSError as err:
        print(
            f"Cannot open UDP port {args.port} on {args.bind} ({err.strerror or err}). Stop the dashboard "
            "(or another receiver) or choose another port with --port."
        )
        return 2

    print(f"Listening on {args.bind}:{sock.getsockname()[1]} for {args.seconds:g} s (Ctrl+C to stop)...\n")
    try:
        senders, elapsed = listen(sock, args.seconds, fmt)
    finally:
        sock.close()

    if not senders:
        print(f"Nothing received in {elapsed:.0f} s. Check:")
        print(f"  - that ibaPDA sends to the IP address of this machine, port {args.port} (with a VPN: the tunnel IP);")
        print("  - that the firewall (Windows: inbound UDP for python.exe) allows this port;")
        print("  - that no other program is listening on this port and that the output module in ibaPDA is active.")
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
        print("Everything fits the format.")
        print(f"For the dashboard: ALMA_UDP_ALLOWED_SOURCES={hosts}")
        return 0
    print("Not all telegrams fit the format. Check length, offsets and byte order in ibaPDA")
    print("and in udp_format.json; with --dump you can see the raw bytes in both byte orders.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
