"""The UDP receiver with real packets over loopback: filter, outcomes, counters and the socket itself."""

import socket
import time

import pytest

from alma.config import ROOT, SIGNALS
from alma.ingest.cache import SampleCache
from alma.ingest.format import MAX_PACKET_BYTES, PacketFormat
from alma.ingest.receiver import UdpReceiver

FORMAT = PacketFormat.load(ROOT / "udp_format.json", required_signals=SIGNALS)
VALUES = {name: float(i) for i, name in enumerate(SIGNALS)}


class Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def processed(receiver: UdpReceiver) -> int:
    stats = receiver.stats()
    return stats["accepted"] + stats["dropped"] + stats["blocked"] + stats["malformed"]


@pytest.fixture
def make():
    started: list[UdpReceiver] = []

    def build(*, allowed_sources=("127.0.0.1",), clock=time.time) -> tuple[UdpReceiver, SampleCache]:
        cache = SampleCache(SIGNALS, retention_s=600, max_samples=10_000, flags=("system_mode",), clock=clock)
        receiver = UdpReceiver(
            FORMAT, cache, bind="127.0.0.1", port=0, allowed_sources=allowed_sources, clock=clock
        )
        receiver.start()
        started.append(receiver)
        return receiver, cache

    yield build
    for receiver in started:
        receiver.stop()


def send(receiver: UdpReceiver, *datagrams: bytes) -> None:
    """Send the datagrams and wait until the receiver has handled them one by one."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        for datagram in datagrams:
            before = processed(receiver)
            sender.sendto(datagram, ("127.0.0.1", receiver.port))
            deadline = time.monotonic() + 2
            while processed(receiver) == before:
                assert time.monotonic() < deadline, "telegram was not processed"
                time.sleep(0.0005)


def test_a_telegram_ends_up_in_the_cache(make):
    receiver, cache = make()
    send(receiver, FORMAT.encode(VALUES))
    stats = receiver.stats()
    assert stats["accepted"] == stats["received"] == 1 and stats["last_packet_age_s"] < 2
    frame = cache.window(0, time.time() + 1, 100)
    assert len(frame) == 1 and list(frame.columns) == SIGNALS
    assert frame.iloc[0]["pressure_bore_skid1"] == pytest.approx(VALUES["pressure_bore_skid1"])


def test_before_any_telegram_there_is_nothing_to_report(make):
    receiver, _ = make()
    stats = receiver.stats()
    assert stats["received"] == 0 and stats["last_packet_age_s"] is None
    assert receiver.running and receiver.port > 0


@pytest.mark.parametrize("allowed", [None, ["127.0.0.1"], ["127.0.0.0/8"], ["10.9.9.9", "127.0.0.1"]])
def test_allowed_sources_let_the_sender_through(make, allowed):
    receiver, cache = make(allowed_sources=allowed)
    send(receiver, FORMAT.encode(VALUES))
    assert receiver.stats()["accepted"] == 1 and len(cache) == 1


@pytest.mark.parametrize("allowed", [[], ["10.9.9.9"], ["10.0.0.0/8"], ["::1"]])
def test_other_senders_are_blocked_before_decoding(make, allowed):
    receiver, cache = make(allowed_sources=allowed)
    send(receiver, FORMAT.encode(VALUES))
    stats = receiver.stats()
    assert stats["blocked"] == 1 and stats["accepted"] == 0 and len(cache) == 0


def test_telegrams_that_do_not_fit_are_counted_and_the_receiver_keeps_going(make):
    receiver, cache = make()
    good = FORMAT.encode(VALUES)
    send(receiver, b"too short", good + b"\x00", b"", bytes(MAX_PACKET_BYTES), good)
    stats = receiver.stats()
    assert stats["malformed"] == 4 and stats["accepted"] == 1 and stats["received"] == 5
    assert len(cache) == 1 and receiver.running


def test_a_value_that_is_not_a_number_is_refused(make):
    receiver, cache = make()
    nan = bytearray(FORMAT.encode(VALUES))
    nan[0:4] = b"\x7f\xc0\x00\x00"  # tower_x = NaN (big endian float32)
    send(receiver, bytes(nan))
    assert receiver.stats()["malformed"] == 1 and len(cache) == 0


def test_a_repeated_timestamp_is_dropped_by_the_cache(make):
    clock = Clock(1_700_000_000.0)
    receiver, cache = make(clock=clock)
    send(receiver, FORMAT.encode(VALUES), FORMAT.encode(VALUES))  # the clock stands still: same timestamp twice
    stats = receiver.stats()
    assert stats["accepted"] == 1 and stats["dropped"] == 1 and stats["received"] == 2
    clock.now += 0.05
    send(receiver, FORMAT.encode(VALUES))
    assert receiver.stats()["accepted"] == 2 and len(cache) == 2


def test_packet_age_follows_the_clock(make):
    clock = Clock(1_700_000_000.0)
    receiver, _ = make(clock=clock)
    send(receiver, FORMAT.encode(VALUES))
    clock.now += 12.34
    assert receiver.stats()["last_packet_age_s"] == pytest.approx(12.3)


def test_stats_have_the_keys_udp_source_reads(make):
    receiver, _ = make()
    assert {"received", "accepted", "dropped", "blocked", "malformed", "last_packet_age_s", "frozen", "loss_percent"} <= set(
        receiver.stats()
    )


def test_a_taken_port_is_reported_in_plain_words():
    blocker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    blocker.bind(("127.0.0.1", 0))
    try:
        cache = SampleCache(SIGNALS, retention_s=60, max_samples=1_000)
        receiver = UdpReceiver(FORMAT, cache, bind="127.0.0.1", port=blocker.getsockname()[1], allowed_sources=None)
        with pytest.raises(RuntimeError, match="UDP port"):
            receiver.start()
        assert not receiver.running
    finally:
        blocker.close()


def test_stop_frees_the_port_and_the_receiver_can_start_again(make):
    receiver, _ = make()
    port = receiver.port
    receiver.stop()
    assert not receiver.running
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as again:
        again.bind(("127.0.0.1", port))  # the port is free
    receiver.stop()  # stopping again is allowed
    receiver.start()
    send(receiver, FORMAT.encode(VALUES))
    assert receiver.stats()["accepted"] == 1


def test_start_twice_is_harmless(make):
    receiver, _ = make()
    port = receiver.port
    receiver.start()
    assert receiver.running and receiver.port == port
