"""alma.udp_check against a real sender over loopback: report, exit code and the byte table."""

import json
import socket
import threading
import time

import pytest

from alma import udp_check
from alma.config import ROOT, SIGNALS
from alma.ingest.format import PacketFormat

FORMAT = PacketFormat.load(ROOT / "udp_format.json", required_signals=SIGNALS)
VALUES = {name: float(i + 1) for i, name in enumerate(SIGNALS)}


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Sender:
    """Sends telegrams in a thread (10 per second) until stopped, also before the tool is listening."""

    def __init__(self, port: int, packet: bytes) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(port, packet), daemon=True)

    def _run(self, port: int, packet: bytes) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            while not self._stop.is_set():
                sock.sendto(packet, ("127.0.0.1", port))
                time.sleep(0.05)

    def __enter__(self) -> "Sender":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


def check(*extra: str, port: int, seconds: float = 1.0) -> int:
    return udp_check.main(["--port", str(port), "--bind", "127.0.0.1", "--seconds", str(seconds), *extra])


def test_matching_telegrams_give_a_clean_report(capsys):
    port = free_port()
    with Sender(port, FORMAT.encode(VALUES)):
        code = check(port=port)
    out = capsys.readouterr().out
    assert code == 0
    assert "Source 127.0.0.1" in out and "Everything fits the format" in out
    assert "ALMA_UDP_ALLOWED_SOURCES=127.0.0.1" in out
    for name in SIGNALS:
        assert name in out


def test_telegrams_of_the_wrong_size_are_reported(capsys):
    port = free_port()
    with Sender(port, bytes(30)):
        code = check(port=port)
    out = capsys.readouterr().out
    assert code == 1
    assert "30 bytes" in out and "expects 36 bytes" in out and "Not all telegrams fit" in out


def test_silence_is_reported_with_hints(capsys):
    code = check(port=free_port(), seconds=0.3)
    out = capsys.readouterr().out
    assert code == 2 and "Nothing received" in out and "firewall" in out


def test_a_busy_port_is_reported(capsys):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as blocker:
        blocker.bind(("127.0.0.1", 0))
        code = check(port=blocker.getsockname()[1], seconds=0.3)
    assert code == 2 and "Cannot open" in capsys.readouterr().out


def test_a_broken_format_still_shows_what_arrives(capsys, tmp_path):
    broken = tmp_path / "format.json"
    broken.write_text(json.dumps({"packet_bytes": 28, "fields": []}), encoding="utf-8")
    port = free_port()
    with Sender(port, FORMAT.encode(VALUES)):
        code = check("--format", str(broken), port=port)
    out = capsys.readouterr().out
    assert code == 1 and "The format is not usable" in out and "Source 127.0.0.1" in out


def test_dump_shows_every_offset_in_both_byte_orders(capsys):
    port = free_port()
    with Sender(port, FORMAT.encode(VALUES)):
        check("--dump", port=port)
    out = capsys.readouterr().out
    assert "FLOAT big" in out and "FLOAT little" in out
    rows = [line for line in out.splitlines() if line.strip().split(" ")[0] in {str(o) for o in range(0, 36, 4)}]
    assert len(rows) == 9
    assert f"{VALUES['tower_x']:g}" in rows[0].split()  # tower_x is at offset 0: the first float (big endian)


def test_dump_table_decodes_floats_and_integers():
    packet = FORMAT.encode(VALUES)
    table = udp_check.dump(packet, step=4)
    first = [line for line in table.splitlines() if line.lstrip().startswith("0 ")][0].split()
    assert float(first[5]) == VALUES["tower_x"]  # offset, four bytes, FLOAT big
    assert f"Telegram of {len(packet)} bytes" in table


def test_bind_must_be_an_ip_address():
    with pytest.raises(SystemExit):
        udp_check.main(["--bind", "not-an-ip"])
