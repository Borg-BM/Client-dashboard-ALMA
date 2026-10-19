"""Pakketverlies schatten uit aankomsttijden (zonder telbare hartslag)."""

import numpy as np
import pytest

from alma.ingest.linkquality import MIN_SAMPLES, estimate_loss_percent

PERIOD = 0.05  # ibaPDA: 20 telegrammen per seconde


def stream(count=1200, removed=()):
    times = np.arange(count) * PERIOD
    return np.delete(times, list(removed))


def test_a_complete_stream_has_no_loss():
    assert estimate_loss_percent(stream()) == pytest.approx(0.0, abs=1e-9)


def test_missing_packets_inside_the_stream_are_counted():
    # 8 van de 100 weg, begin en einde blijven: verwacht 100, ontvangen 92
    assert estimate_loss_percent(stream(100, removed=range(10, 90, 10))) == pytest.approx(8.0)


def test_an_outage_counts_as_loss():
    times = stream(1200)  # 60 s
    outage = (times >= 20) & (times < 30)  # 10 s uitval: Starlink
    assert estimate_loss_percent(times[~outage]) == pytest.approx(100 * 10 / 60, abs=0.5)


def test_jitter_is_not_mistaken_for_loss():
    rng = np.random.default_rng(1)
    jittered = np.sort(stream(1200) + rng.normal(0, 0.010, 1200))  # 10 ms jitter
    assert estimate_loss_percent(jittered) < 1.0


def test_random_loss_is_estimated_within_a_point():
    rng = np.random.default_rng(2)
    keep = rng.random(2400) > 0.10
    keep[[0, -1]] = True
    assert estimate_loss_percent(stream(2400)[keep]) == pytest.approx(10.0, abs=1.5)


def test_too_few_packets_say_nothing():
    assert estimate_loss_percent(stream(MIN_SAMPLES - 1)) is None
    assert estimate_loss_percent(stream(MIN_SAMPLES)) is not None


@pytest.mark.parametrize(
    "times",
    [np.zeros(50), np.full(50, 3.0), np.r_[np.arange(49) * PERIOD, np.nan], np.r_[np.arange(49) * PERIOD, np.inf]],
    ids=["all-equal-zero", "all-equal", "nan", "inf"],
)
def test_unusable_input_gives_none_not_an_error(times):
    assert estimate_loss_percent(times) is None


def test_accepts_any_sequence_of_numbers():
    assert estimate_loss_percent(list(stream(100))) == pytest.approx(0.0, abs=1e-9)
