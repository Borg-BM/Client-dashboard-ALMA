"""Tests for the sample cache (alma.ingest.cache): ring, retention, reading with downsampling and concurrent use.

Everything is deterministic: a set clock, a fixed seed, no network and no real iba.
"""

import gc
import math
import random
import sys
import threading
import time
import tracemalloc
from dataclasses import FrozenInstanceError, asdict
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from alma import analysis
from alma.config import SIGNALS, CylinderGeometry
from alma.ingest import cache as cache_module
from alma.ingest.cache import CacheStats, SampleCache
from alma.sources.mock import TOWER_FREQ_HZ, MockSource

NAMES = ["a", "b", "flag"]
FLAG_COLUMNS = {2}  # position of "flag" in NAMES
T0 = 1_700_000_000.0
FOREVER = 1e9  # retention that does not come into play in most tests


class Clock:
    """Manually advanced clock instead of time.time, so that the retention is deterministic."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def row(i: float) -> dict:
    """A row whose values reveal the index: this shows which values belong to which row."""
    return {"a": float(i), "b": 10.0 * i, "flag": float(i % 2)}


def make(max_samples: int = 100, *, retention_s: float = FOREVER, clock=None, flags=("flag",), signals=NAMES, **kwargs):
    return SampleCache(
        signals, retention_s=retention_s, max_samples=max_samples, flags=flags, clock=clock or Clock(), **kwargs
    )


def fill(cache: SampleCache, times) -> None:
    for t in times:
        assert cache.append(t, row(t)) is True, t


def epoch(df: pd.DataFrame) -> np.ndarray:
    """Times of the index as float seconds, converted from the nanoseconds without losing precision."""
    ns = df.index.as_unit("ns").asi8
    return ns // 1_000_000_000 + (ns % 1_000_000_000) / 1e9


def assert_stamps(df: pd.DataFrame, expected, atol: float = 1e-9) -> None:
    np.testing.assert_allclose(epoch(df), expected, rtol=0, atol=atol)


def reference_buckets(t: np.ndarray, v: np.ndarray, max_points: int, flag_cols=FLAG_COLUMNS):
    """Reference for the downsampling, deliberately using a plain loop and np.mean/np.max instead of reduceat."""
    if len(t) <= max_points:
        return t, v
    k = math.ceil(len(t) / max_points)
    out_t, out_v = [], []
    for i in range(0, len(t), k):
        chunk = v[i : i + k]
        out_t.append(np.mean(t[i : i + k]))
        out_v.append([chunk[:, j].max() if j in flag_cols else chunk[:, j].mean() for j in range(v.shape[1])])
    return np.array(out_t), np.array(out_v)


def run_with_timeout(func, timeout: float = 5.0):
    """Run func in a thread: a hanging call (deadlock) makes the test fail instead of hanging."""
    box: dict = {}

    def target():
        try:
            box["result"] = func()
        except BaseException as err:  # noqa: BLE001 - pass on to the test thread
            box["error"] = err

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), "the call hangs (deadlock?)"
    if "error" in box:
        raise box["error"]
    return box["result"]


# ---- configuration ---------------------------------------------------------------------------------

BAD_CONFIGS = [
    pytest.param({"signals": []}, "signals", id="no-signals"),
    pytest.param({"signals": ["a", "b", "a"]}, "unique", id="duplicate-signals"),
    pytest.param({"signals": "abc"}, "signals", id="signals-as-one-string"),
    pytest.param({"retention_s": 0}, "retention_s", id="retention-zero"),
    pytest.param({"retention_s": -5.0}, "retention_s", id="retention-negative"),
    pytest.param({"retention_s": math.inf}, "retention_s", id="retention-infinite"),
    pytest.param({"retention_s": math.nan}, "retention_s", id="retention-nan"),
    pytest.param({"max_samples": 1}, "max_samples", id="capacity-one"),
    pytest.param({"max_samples": 0}, "max_samples", id="capacity-zero"),
    pytest.param({"max_samples": -10}, "max_samples", id="capacity-negative"),
    pytest.param({"max_samples": 10.5}, "max_samples", id="capacity-fractional"),
    pytest.param({"clock_step_back_s": 0}, "clock_step_back_s", id="step-back-zero"),
    pytest.param({"clock_step_back_s": -1.0}, "clock_step_back_s", id="step-back-negative"),
    pytest.param({"clock_step_back_s": math.nan}, "clock_step_back_s", id="step-back-nan"),
    pytest.param({"flags": ["nope"]}, "flags", id="flag-that-is-no-signal"),
    pytest.param({"flags": ["a", "nope"]}, "nope", id="one-flag-too-many"),
]


@pytest.mark.parametrize(("override", "fragment"), BAD_CONFIGS)
def test_constructor_rejects_an_invalid_configuration(override, fragment):
    kwargs = {"signals": NAMES, "retention_s": 60.0, "max_samples": 10} | override
    signals = kwargs.pop("signals")
    with pytest.raises(ValueError, match=fragment):
        SampleCache(signals, **kwargs)


def test_constructor_accepts_the_smallest_valid_configuration():
    cache = SampleCache(["only"], retention_s=0.001, max_samples=2, flags=["only"], clock=Clock(), clock_step_back_s=1e-9)
    assert cache.append(1.0, {"only": 1.0}) and cache.append(2.0, {"only": 0.0})
    assert cache.append(3.0, {"only": 1.0})
    assert len(cache) == 2  # a ring of two: the oldest row is gone


# ---- shape of the result -----------------------------------------------------------------------


def test_window_gives_a_float_frame_with_a_utc_index_in_signal_order():
    cache = make(signals=["pressure", "tower_x", "flag"])
    stamps = [T0 + 0.05 * i for i in range(5)]
    for i, t in enumerate(stamps):
        # an int, a bool and a numpy float must be accepted just like a float
        assert cache.append(t, {"flag": True, "tower_x": i, "pressure": np.float32(1.5)})

    df = cache.window(T0, T0 + 1, 100)

    assert list(df.columns) == ["pressure", "tower_x", "flag"]
    assert (df.dtypes == np.float64).all()
    assert isinstance(df.index, pd.DatetimeIndex) and str(df.index.tz) == "UTC"
    assert_stamps(df, stamps)
    assert df["tower_x"].tolist() == [0, 1, 2, 3, 4] and df["pressure"].eq(1.5).all() and df["flag"].eq(1.0).all()


def test_columns_follow_the_constructor_order_and_extra_keys_are_ignored():
    cache = make()
    # different order than NAMES, and keys that are not a signal (also with an unusable value)
    assert cache.append(1.0, {"flag": 1.0, "zzz": math.nan, "b": 20.0, "a": 2.0, "extra": "text"}) is True
    df = cache.window(0, 2, 10)
    assert list(df.columns) == NAMES
    assert df.iloc[0].tolist() == [2.0, 20.0, 1.0]
    assert cache.stats().dropped_invalid == 0


def test_epoch_times_survive_the_trip_through_the_frame():
    cache = make(1_000)
    stamps = T0 + np.cumsum(np.random.default_rng(7).uniform(0.05, 0.3, 500))
    fill(cache, stamps.tolist())
    assert_stamps(cache.window(0, 2e9, 1_000), stamps)


def test_times_before_1970_and_at_the_edge_of_the_pandas_range_convert_correctly():
    stamps = [-9_223_372_036.0, -1e9 - 0.5, -1.25, 0.0, 0.5, 9_223_372_035.5, 9_223_372_036.0]
    cache = make(retention_s=1e12)  # the default retention (1e9 s) would drop the oldest rows when reading
    fill(cache, stamps)  # the extreme limits still belong in; a fraction of a second further does not (see below)
    df = cache.window(-math.inf, math.inf, 100)
    assert_stamps(df, stamps)
    assert df.index[2] == pd.Timestamp("1969-12-31 23:59:58.75", tz="UTC")  # negative times with a fraction
    assert df.index[3] == pd.Timestamp("1970-01-01", tz="UTC")
    assert cache.append(9_223_372_036.5, row(1)) is False and cache.stats().dropped_invalid == 1


# ---- empty cache -----------------------------------------------------------------------------------


def test_empty_cache_has_no_rows_and_no_extremes():
    cache = make()
    assert len(cache) == 0
    assert cache.stats() == CacheStats(
        rows=0, capacity=100, oldest=None, newest=None, appended=0,
        dropped_out_of_order=0, dropped_invalid=0, overwritten=0, clock_resets=0,
    )  # fmt: skip
    for bounds in [(-math.inf, math.inf), (0, 1), (5, 1)]:
        df = cache.window(*bounds, 10)
        assert df.empty and list(df.columns) == NAMES


def test_every_kind_of_empty_window_has_the_types_of_a_filled_one():
    filled = make()
    fill(filled, [T0 + i for i in range(5)])
    reference = filled.window(0, 2e9, 10)
    expired = make(retention_s=10, clock=Clock(T0 + 1_000))
    fill(expired, [T0 + i for i in range(5)])

    empties = {
        "empty cache": make().window(0, 1, 10),
        "start after end": filled.window(5, 1, 10),
        "no row in range": filled.window(T0 + 100, T0 + 200, 10),
        "everything past the retention": expired.window(0, 2e9, 10),
    }
    for name, df in empties.items():
        assert df.empty, name
        assert isinstance(df.index, pd.DatetimeIndex) and str(df.index.tz) == "UTC", name
        assert df.index.dtype == reference.index.dtype, name  # pandas 3 would give an empty window another unit
        assert list(df.columns) == NAMES and (df.dtypes == reference.dtypes).all(), name


# ---- ring -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("capacity", [2, 3, 4, 7])
def test_ring_keeps_only_the_newest_rows_when_full(capacity):
    cache = make(capacity)
    for n in range(1, 4 * capacity + 1):
        assert cache.append(float(n), row(n)) is True
        kept = [float(i) for i in range(max(1, n - capacity + 1), n + 1)]

        df = cache.window(-math.inf, math.inf, 1_000)
        assert_stamps(df, kept)
        assert df["a"].tolist() == kept
        assert df["b"].tolist() == [10 * t for t in kept]  # the values sit with their own row
        stats = cache.stats()
        assert len(cache) == stats.rows == len(kept)
        assert (stats.oldest, stats.newest) == (kept[0], kept[-1])
        assert stats.appended == n and stats.overwritten == n - len(kept)


@pytest.mark.parametrize("rows", range(5, 12))
def test_windows_across_the_wrap_point_return_exactly_the_matching_rows(rows):
    cache = make(5)  # after 5..11 rows the ring sits at every possible write position, mostly wrapped
    fill(cache, [float(t) for t in range(1, rows + 1)])
    kept = [float(t) for t in range(max(1, rows - 4), rows + 1)]
    edges = sorted({t + d for t in kept for d in (-0.5, 0.0, 0.5)} | {0.0, 100.0})
    for start in edges:
        for end in edges:
            assert_stamps(cache.window(start, end, 100), [t for t in kept if start <= t <= end])


def test_a_flood_of_rows_neither_grows_nor_copies_the_ring():
    capacity, flood = 10_000, 8_000
    cache = make(capacity)  # the arrays together are ~320 kB: a copy or growth is therefore clearly visible
    tracing_already = tracemalloc.is_tracing()
    if not tracing_already:
        tracemalloc.start()
    try:
        fill(cache, [float(i) for i in range(capacity + 10)])  # ring full and wrapped: everything needed exists now
        gc.collect()
        before = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        for i in range(capacity + 10, capacity + 10 + flood):
            cache.append(float(i), row(i))
        gc.collect()
        after, peak = tracemalloc.get_traced_memory()
    finally:
        if not tracing_already:
            tracemalloc.stop()
    assert len(cache) == capacity and cache.stats().overwritten == flood + 10
    # Per stored row ~100 bytes would soon be added (megabytes in total); this is only the noise of the counters.
    assert after - before < 32 * 1024  # no growth
    assert peak - before < 32 * 1024  # and no temporary copy of the ring during append() either


# ---- window edges ----------------------------------------------------------------------------------


def test_window_bounds_are_inclusive():
    cache = make()
    fill(cache, [10, 11, 12, 13, 14])
    assert_stamps(cache.window(11, 13, 100), [11, 12, 13])
    assert_stamps(cache.window(11, 11, 100), [11])
    assert_stamps(cache.window(14, 99, 100), [14])
    assert_stamps(cache.window(-99, 10, 100), [10])
    assert_stamps(cache.window(10.5, 10.99, 100), [])
    assert_stamps(cache.window(12.5, 12.5, 100), [])


@pytest.mark.parametrize(
    ("start", "end"), [(13, 11), (math.nan, 13), (11, math.nan), (math.nan, math.nan), (math.inf, -math.inf)]
)
def test_window_without_a_valid_range_is_empty(start, end):
    cache = make()
    fill(cache, [10, 11, 12, 13])
    df = cache.window(start, end, 100)
    assert df.empty and list(df.columns) == NAMES


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_window_rejects_max_points_below_one(bad):
    cache = make()
    with pytest.raises(ValueError, match="max_points"):
        cache.window(0, 10, bad)  # empty cache
    fill(cache, [1, 2, 3])
    with pytest.raises(ValueError, match="max_points"):
        cache.window(0, 10, bad)
    with pytest.raises(ValueError, match="max_points"):
        cache.window(10, 0, bad)  # even if the window would be empty anyway: this is an error in the call


# ---- retention -------------------------------------------------------------------------------------


def test_rows_older_than_the_retention_are_left_out():
    clock = Clock(110.0)
    cache = make(retention_s=5, clock=clock)
    fill(cache, range(100, 111))

    assert_stamps(cache.window(0, 1e9, 100), [105, 106, 107, 108, 109, 110])  # t >= 110 - 5, bound included
    clock.now = 110.5
    assert_stamps(cache.window(0, 1e9, 100), [106, 107, 108, 109, 110])  # the clock is read again on every call
    clock.now = 110.0
    assert len(cache.window(0, 1e9, 100)) == 6
    clock.now = 115.0
    assert_stamps(cache.window(0, 1e9, 100), [110])  # 115 - 5 = 110: included
    clock.now = 115.001
    assert_stamps(cache.window(0, 1e9, 100), [])


def test_retention_only_applies_when_reading():
    clock = Clock(1_000.0)
    cache = make(retention_s=5, clock=clock)
    fill(cache, range(100, 111))  # all of them far beyond the retention
    assert cache.window(0, 1e9, 100).empty
    assert len(cache) == 11 and cache.stats().rows == 11  # they are still there: only the ring determines what stays
    clock.now = 108.0  # the clock can also go back (for example after a clock step)
    assert_stamps(cache.window(0, 1e9, 100), [103, 104, 105, 106, 107, 108, 109, 110])


def test_start_and_retention_both_limit_the_window():
    clock = Clock(110.0)
    cache = make(retention_s=5, clock=clock)
    fill(cache, range(100, 111))
    assert_stamps(cache.window(108, 1e9, 100), [108, 109, 110])  # start is stricter than the retention
    assert_stamps(cache.window(100, 1e9, 100), [105, 106, 107, 108, 109, 110])  # the retention is stricter
    assert_stamps(cache.window(100, 107, 100), [105, 106, 107])
    assert_stamps(cache.window(100, 104.9, 100), [])  # the end lies before the retention bound


# ---- downsampling ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rows", "max_points"),
    [(1_000, 64), (1_200, 100), (1_000, 999), (1_000, 1), (101, 100), (7, 3), (50, 49), (1_000, 333), (10, 6), (2, 1)],
)
def test_downsampling_matches_a_numpy_reference(rows, max_points):
    rng = np.random.default_rng(42)
    t = T0 + np.cumsum(rng.uniform(0.04, 0.06, rows))  # irregular but strictly increasing
    v = np.column_stack([rng.normal(size=rows), 500 + 100 * rng.normal(size=rows), rng.random(rows) < 0.1]).astype(float)
    cache = make(rows + 5)
    for stamp, values in zip(t.tolist(), v.tolist(), strict=True):
        assert cache.append(stamp, dict(zip(NAMES, values, strict=True)))

    df = cache.window(-math.inf, math.inf, max_points)

    ref_t, ref_v = reference_buckets(t, v, max_points)
    k = math.ceil(rows / max_points) if rows > max_points else 1
    assert len(df) == len(ref_t) == math.ceil(rows / k) and len(df) <= max_points
    np.testing.assert_allclose(epoch(df), ref_t, rtol=0, atol=1e-6)
    np.testing.assert_allclose(df.to_numpy(), ref_v, rtol=1e-12, atol=1e-12)
    assert (np.diff(epoch(df)) > 0).all()
    assert (df.dtypes == np.float64).all()


@pytest.mark.parametrize("rows", range(1, 41))
def test_downsampled_result_never_exceeds_max_points_for_any_small_combination(rows):
    cache = make(100)
    fill(cache, range(1, rows + 1))
    t = np.arange(1.0, rows + 1)
    v = np.column_stack([t, 10 * t, t % 2])  # same values as row()
    for max_points in range(1, rows + 3):
        df = cache.window(0, 1e9, max_points)
        k = math.ceil(rows / max_points) if rows > max_points else 1
        ref_t, ref_v = reference_buckets(t, v, max_points)
        assert len(df) == len(ref_t) == math.ceil(rows / k) <= max_points
        assert_stamps(df, ref_t)
        np.testing.assert_allclose(df.to_numpy(), ref_v, rtol=1e-12)


def test_rows_come_back_unchanged_when_they_fit_in_max_points():
    cache = make()
    fill(cache, range(1, 11))
    for max_points in (10, 11, 1_000):  # ten rows are not "more than" ten
        df = cache.window(0, 100, max_points)
        assert_stamps(df, range(1, 11))
        assert df["a"].tolist() == list(range(1, 11))
    assert len(cache.window(0, 100, 9)) == 5  # k = 2


def test_bucket_time_is_the_mean_of_the_times_in_the_bucket():
    cache = make()
    fill(cache, [0, 1, 3, 10, 11, 20])  # irregular
    df = cache.window(-1, 100, 3)  # k = 2: (0, 1) (3, 10) (11, 20)
    assert_stamps(df, [0.5, 6.5, 15.5])
    assert df["a"].tolist() == [0.5, 6.5, 15.5]
    assert df["b"].tolist() == [5.0, 65.0, 155.0]


def test_the_last_bucket_may_be_shorter():
    cache = make()
    fill(cache, range(1, 11))  # ten rows, max_points 4: k = 3, so 3 + 3 + 3 + 1
    df = cache.window(0, 100, 4)
    assert_stamps(df, [2, 5, 8, 10])
    assert df["a"].tolist() == [2, 5, 8, 10]


def test_flag_columns_keep_a_short_pulse_that_analog_columns_average_away():
    cache = SampleCache(["level", "on"], retention_s=FOREVER, max_samples=2_000, flags=["on"], clock=Clock())
    for i in range(1_000):
        pulse = 1.0 if i == 500 else 0.0  # a pulse of a single row
        assert cache.append(float(i), {"level": pulse, "on": pulse})

    df = cache.window(0, 1_000, 10)  # k = 100

    assert df["on"].tolist() == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  # maximum: the pulse is still there
    assert df["level"].max() == pytest.approx(0.01)  # the same signal as an analog channel is averaged down to 1/100


def test_every_flag_column_uses_max_wherever_it_sits_and_whatever_its_sign():
    cache = SampleCache(
        ["f1", "x", "f2", "y"], retention_s=FOREVER, max_samples=100, flags=["f1", "f2"], clock=Clock()
    )
    fill_rows = [
        (0, {"f1": 0.0, "x": 1.0, "f2": -3.0, "y": 10.0}),
        (1, {"f1": 1.0, "x": 3.0, "f2": -1.0, "y": 30.0}),
        (2, {"f1": 0.0, "x": 5.0, "f2": -2.0, "y": 50.0}),
        (3, {"f1": 0.0, "x": 7.0, "f2": -9.0, "y": 70.0}),
    ]
    for t, values in fill_rows:
        assert cache.append(t, values)
    df = cache.window(0, 10, 2)  # two buckets of two rows
    assert df["f1"].tolist() == [1.0, 0.0]
    assert df["f2"].tolist() == [-1.0, -2.0]  # the maximum, not the largest absolute value or the mean
    assert df["x"].tolist() == [2.0, 6.0] and df["y"].tolist() == [20.0, 60.0]


def test_flags_are_not_touched_when_nothing_is_downsampled():
    cache = make()
    fill(cache, range(1, 6))
    assert cache.window(0, 10, 100)["flag"].tolist() == [1, 0, 1, 0, 1]


# ---- time order -------------------------------------------------------------------------


def test_out_of_order_rows_are_dropped_and_counted():
    cache = make()
    assert cache.append(10.0, row(10)) is True
    assert cache.append(9.0, row(9)) is False
    assert cache.append(9.999, row(99)) is False
    assert cache.stats().dropped_out_of_order == 2
    assert cache.append(11.0, row(11)) is True  # after that it simply works again
    df = cache.window(0, 100, 10)
    assert_stamps(df, [10, 11])
    assert df["a"].tolist() == [10, 11]  # the discarded rows (a=9, a=99) did not sneak in between


def test_equal_timestamp_is_dropped_and_the_first_row_wins():
    cache = make()
    assert cache.append(10.0, {"a": 1.0, "b": 2.0, "flag": 0.0}) is True
    assert cache.append(10.0, {"a": 9.0, "b": 9.0, "flag": 1.0}) is False
    stats = cache.stats()
    assert (stats.dropped_out_of_order, stats.clock_resets, stats.rows) == (1, 0, 1)
    assert cache.window(0, 100, 10).iloc[0].tolist() == [1.0, 2.0, 0.0]


def test_small_clock_step_back_counts_as_out_of_order():
    cache = make(clock_step_back_s=5.0)
    assert cache.append(100.0, row(100)) is True
    for t in (99.0, 96.0, 95.0):  # 95 is exactly 5 s back: only "more than" 5 s counts as a clock step
        assert cache.append(t, row(t)) is False
    stats = cache.stats()
    assert (stats.dropped_out_of_order, stats.clock_resets, stats.rows) == (3, 0, 1)
    assert_stamps(cache.window(0, 1_000, 10), [100])


def test_large_clock_step_back_clears_the_cache_and_starts_over():
    cache = make(clock_step_back_s=5.0)
    fill(cache, [98, 99, 100])
    assert cache.append(94.5, row(94.5)) is True  # 5.5 s back: the system clock was set back

    stats = cache.stats()
    assert (stats.rows, stats.oldest, stats.newest) == (1, 94.5, 94.5)
    assert (stats.clock_resets, stats.dropped_out_of_order, stats.appended) == (1, 0, 4)
    assert_stamps(cache.window(0, 1_000, 10), [94.5])  # the old rows are gone
    fill(cache, [95, 96])  # and reception just continues
    assert_stamps(cache.window(0, 1_000, 10), [94.5, 95, 96])


def test_clock_step_back_threshold_is_configurable():
    cache = make(clock_step_back_s=0.5)
    cache.append(100.0, row(100))
    assert cache.append(99.5, row(99.5)) is False  # exactly at the threshold
    assert cache.stats().clock_resets == 0
    assert cache.append(99.4, row(99.4)) is True
    assert cache.stats().clock_resets == 1


def test_clock_reset_on_a_full_ring_leaves_one_row_and_a_working_ring():
    cache = make(4, clock_step_back_s=5.0)
    fill(cache, range(100, 110))  # the ring is full and wrapped, six rows overwritten
    assert cache.append(50.0, row(50)) is True
    assert len(cache) == 1 and cache.stats().overwritten == 6  # the reset does not count as overwriting

    fill(cache, [51, 52, 53, 54, 55])  # the ring fills up again and wraps again
    assert_stamps(cache.window(0, 1_000, 10), [52, 53, 54, 55])
    assert cache.stats().overwritten == 8


def test_every_large_step_back_is_a_separate_reset():
    cache = make(clock_step_back_s=5.0)
    fill(cache, [1_000, 1_001])
    for resets, t in enumerate([900, 800, 700], start=1):
        assert cache.append(float(t), row(t)) is True
        assert cache.stats().clock_resets == resets and len(cache) == 1
    assert_stamps(cache.window(0, 2_000, 10), [700])


def test_clear_empties_the_cache_and_accepts_any_timestamp_afterwards():
    cache = make()
    fill(cache, range(100, 106))
    cache.clear()

    assert len(cache) == 0 and cache.window(0, 1_000, 10).empty
    stats = cache.stats()
    assert (stats.rows, stats.oldest, stats.newest) == (0, None, None)
    assert stats.appended == 6  # the counters cover the lifetime of the cache
    assert cache.append(1.0, row(1)) is True  # far before the previous newest row: not out-of-order, no clock reset
    stats = cache.stats()
    assert (stats.rows, stats.dropped_out_of_order, stats.clock_resets) == (1, 0, 0)


# ---- bad data ---------------------------------------------------------------------------------

INVALID_TIMESTAMPS = [
    pytest.param(math.nan, id="nan"),
    pytest.param(np.nan, id="numpy-nan"),
    pytest.param(math.inf, id="plus-inf"),
    pytest.param(-math.inf, id="minus-inf"),
    pytest.param(None, id="none"),
    pytest.param("100", id="text"),
    pytest.param(object(), id="object"),
    pytest.param([1.0], id="list"),
    pytest.param(1 + 2j, id="complex"),
    pytest.param(10**400, id="int-too-big-for-a-float"),
    pytest.param(1e12, id="year-33658-not-representable-in-pandas"),
    pytest.param(-1e12, id="before-the-pandas-range"),
    pytest.param(9_300_000_000.0, id="just-beyond-the-pandas-range"),
]


@pytest.mark.parametrize("timestamp", INVALID_TIMESTAMPS)
def test_invalid_timestamp_is_dropped_without_raising(timestamp):
    cache = make()
    cache.append(100.0, row(100))
    assert cache.append(timestamp, row(1)) is False
    stats = cache.stats()
    assert (stats.dropped_invalid, stats.rows, stats.newest, stats.appended) == (1, 1, 100.0, 1)
    assert_stamps(cache.window(-math.inf, math.inf, 10), [100])  # and the rest remains readable


INVALID_ROWS = [
    pytest.param({"a": 1.0, "b": 2.0}, id="missing-signal"),
    pytest.param({}, id="empty-mapping"),
    pytest.param({"a": math.nan, "b": 2.0, "flag": 0.0}, id="nan-first"),
    pytest.param({"a": 1.0, "b": 2.0, "flag": math.nan}, id="nan-last"),
    pytest.param({"a": 1.0, "b": math.inf, "flag": 0.0}, id="plus-inf"),
    pytest.param({"a": 1.0, "b": -math.inf, "flag": 0.0}, id="minus-inf"),
    pytest.param({"a": 1.0, "b": None, "flag": 0.0}, id="none"),
    pytest.param({"a": 1.0, "b": "2.5", "flag": 0.0}, id="numeric-text"),
    pytest.param({"a": 1.0, "b": "text", "flag": 0.0}, id="text"),
    pytest.param({"a": 1.0, "b": [2.0], "flag": 0.0}, id="list"),
    pytest.param({"a": 1.0, "b": 10**400, "flag": 0.0}, id="int-too-big-for-a-float"),
    pytest.param(None, id="values-is-none"),
    pytest.param(42, id="values-is-a-number"),
    pytest.param("abc", id="values-is-text"),
    pytest.param([1.0, 2.0, 3.0], id="values-is-a-list"),
    pytest.param(np.array([1.0, 2.0, 3.0]), id="values-is-an-array"),
]


@pytest.mark.parametrize("values", INVALID_ROWS)
def test_invalid_values_are_dropped_without_raising_and_leave_the_cache_untouched(values):
    cache = make()
    fill(cache, [10, 11])
    before = cache.window(0, 100, 10)

    assert cache.append(12.0, values) is False

    stats = cache.stats()
    assert (stats.dropped_invalid, stats.rows, stats.appended, stats.newest) == (1, 2, 2, 11.0)
    pd.testing.assert_frame_equal(cache.window(0, 100, 10), before)  # not stored halfway either
    assert cache.append(12.0, row(12)) is True  # the same time is still fine afterwards


def test_an_invalid_row_never_counts_as_out_of_order_or_triggers_a_reset():
    cache = make(clock_step_back_s=5.0)
    fill(cache, [100, 101])
    assert cache.append(50.0, {"a": math.nan, "b": 1.0, "flag": 0.0}) is False  # 51 s back, but invalid
    assert cache.append(100.5, {"a": 1.0}) is False  # also invalid, and not increasing
    stats = cache.stats()
    assert (stats.dropped_invalid, stats.dropped_out_of_order, stats.clock_resets, stats.rows) == (2, 0, 0, 2)


# ---- copy semantics -----------------------------------------------------------------------------


def test_changing_a_returned_frame_does_not_change_the_cache():
    cache = make()
    fill(cache, range(1, 6))
    for limit in (100, 2):  # without and with downsampling
        frame = cache.window(0, 10, limit)
        expected = frame.copy()
        frame["a"] = -1.0
        frame.iloc[0] = 99.0
        frame.index = frame.index + pd.Timedelta(days=1)
        pd.testing.assert_frame_equal(cache.window(0, 10, limit), expected)

    one, two = cache.window(0, 10, 100), cache.window(0, 10, 100)
    one.iloc[:, :] = 0.0  # two consecutive results do not share anything with each other either
    assert two["a"].tolist() == [1, 2, 3, 4, 5]


def test_a_returned_frame_does_not_change_when_the_ring_overwrites_its_rows():
    cache = make(4)
    fill(cache, range(1, 5))
    frame = cache.window(0, 100, 100)
    expected = frame.copy()
    fill(cache, range(5, 40))  # every position in the ring is overwritten several times
    pd.testing.assert_frame_equal(frame, expected)


# ---- counters --------------------------------------------------------------------------------------


def test_stats_count_every_kind_of_drop_separately():
    cache = make(3, clock_step_back_s=5.0)
    fill(cache, [10, 11, 12, 13])  # four rows in a ring of three: one overwritten
    cache.append(13.0, row(13))  # equal
    cache.append(12.0, row(12))  # older
    cache.append(math.nan, row(1))
    cache.append(14.0, {"a": 1.0})
    cache.append(14.0, {"a": math.inf, "b": 1.0, "flag": 0.0})
    cache.append(2.0, row(2))  # far back: clock reset, and this row is then the only one

    assert cache.stats() == CacheStats(
        rows=1, capacity=3, oldest=2.0, newest=2.0, appended=5,
        dropped_out_of_order=2, dropped_invalid=3, overwritten=1, clock_resets=1,
    )  # fmt: skip


def test_stats_are_a_frozen_snapshot():
    cache = make()
    fill(cache, [1, 2])
    snapshot = cache.stats()
    with pytest.raises(FrozenInstanceError):
        snapshot.rows = 99
    fill(cache, [3])
    assert snapshot.rows == 2 and cache.stats().rows == 3  # an old result does not change along
    assert list(asdict(snapshot)) == [
        "rows", "capacity", "oldest", "newest", "appended",
        "dropped_out_of_order", "dropped_invalid", "overwritten", "clock_resets",
    ]  # fmt: skip
    # plain floats, no numpy types: the API serializes this to JSON as is
    assert type(snapshot.oldest) is float and type(snapshot.newest) is float


# ---- random sequences against a reference model ----------------------------------------------


def test_random_operations_match_a_reference_model():
    rng = random.Random(20240601)
    capacity, retention = 37, 8.0  # the ring spans ~11 s: the retention must really cut into it
    clock = Clock(T0)
    cache = SampleCache(NAMES, retention_s=retention, max_samples=capacity, flags=["flag"], clock=clock)

    kept: list[tuple[float, list[float]]] = []  # model: the last `capacity` accepted rows
    appended = invalid = out_of_order = 0
    t = T0
    compared = downsampled = 0
    for _ in range(2_000):
        action = rng.random()
        if action < 0.50:  # a valid, increasing row
            t += rng.uniform(0.01, 0.6)
            values = [rng.uniform(-50, 50), rng.uniform(0, 1_000), float(rng.random() < 0.15)]
            assert cache.append(t, dict(zip(NAMES, values, strict=True))) is True
            kept.append((t, values))
            del kept[:-capacity]
            appended += 1
            clock.now = t + rng.uniform(-1.0, 3.0)  # the clock more or less keeps up
        elif action < 0.58:  # an invalid row
            stamp, values = rng.choice(
                [
                    (math.nan, row(1)),
                    (t + 1, {"a": math.nan, "b": 1.0, "flag": 0.0}),
                    (t + 1, {"a": 1.0, "b": 2.0}),
                    (t + 1, {"a": 1.0, "b": math.inf, "flag": 0.0}),
                    (None, row(1)),
                    (t + 1, None),
                ]
            )
            assert cache.append(stamp, values) is False
            invalid += 1
        elif action < 0.64 and kept:  # not increasing, but within the clock step threshold (5 s)
            stamp = kept[-1][0] - rng.choice([0.0, rng.uniform(0.0, 4.9)])
            assert cache.append(stamp, row(1)) is False
            out_of_order += 1
        else:  # read
            start = t + rng.uniform(-60.0, 5.0)
            end = start + rng.uniform(-5.0, 70.0)
            limit = rng.choice([1, 2, 3, 7, 20, 100])
            chosen = [(ts, v) for ts, v in kept if start <= ts <= end and ts >= clock.now - retention]
            ref_t = np.array([ts for ts, _ in chosen])
            ref_v = np.array([v for _, v in chosen]).reshape(len(chosen), 3)
            ref_t, ref_v = reference_buckets(ref_t, ref_v, limit)

            df = cache.window(start, end, limit)

            assert len(df) == len(ref_t) and len(df) <= limit
            np.testing.assert_allclose(epoch(df), ref_t, rtol=0, atol=1e-6)
            np.testing.assert_allclose(df.to_numpy().reshape(len(df), 3), ref_v, rtol=1e-12, atol=1e-9)
            assert (np.diff(epoch(df)) > 0).all()
            compared += len(df) > 0
            downsampled += len(chosen) > limit

    stats = cache.stats()
    assert (stats.rows, stats.appended, stats.dropped_invalid, stats.dropped_out_of_order) == (
        len(kept), appended, invalid, out_of_order,
    )  # fmt: skip
    assert stats.overwritten == appended - len(kept) and stats.clock_resets == 0
    assert (stats.oldest, stats.newest) == (kept[0][0], kept[-1][0])
    # A test that compares nothing proves nothing: the sequence must really exercise this.
    assert compared > 200 and downsampled > 60 and len(kept) == capacity


# ---- lock and threads ------------------------------------------------------------------------------


def test_window_reads_the_clock_without_holding_the_lock():
    holder: list[SampleCache] = []

    def clock() -> float:
        return float(len(holder[0]))  # len() takes the lock: if window() still held it, this would hang forever

    cache = SampleCache(NAMES, retention_s=FOREVER, max_samples=10, clock=clock)
    holder.append(cache)
    fill(cache, [1, 2, 3])
    assert_stamps(run_with_timeout(lambda: cache.window(0, 10, 10)), [1, 2, 3])


class SlowFrames:
    """Takes the place of `pd` in the cache module and makes the reader pause while building the DataFrame."""

    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        self.entered, self.release = entered, release

    def __getattr__(self, name):
        return getattr(pd, name)

    def DataFrame(self, *args, **kwargs):  # noqa: N802 - same as pd.DataFrame
        if threading.current_thread().name == "reader":
            self.entered.set()
            assert self.release.wait(10), "the test never released the reader"
        return pd.DataFrame(*args, **kwargs)


def test_the_frame_is_built_outside_the_lock(monkeypatch):
    cache = make()
    fill(cache, [1, 2, 3])
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(cache_module, "pd", SlowFrames(entered, release))

    result: list = []
    reader = threading.Thread(target=lambda: result.append(cache.window(0, 10, 10)), name="reader", daemon=True)
    reader.start()
    try:
        assert entered.wait(5), "window() no longer builds its DataFrame with pd.DataFrame; adjust this test"
        # The reader is now in the middle of the pandas work. A writer must be able to get past it.
        appended: list = []
        writer = threading.Thread(target=lambda: appended.append(cache.append(4.0, row(4))), daemon=True)
        writer.start()
        writer.join(2)
        blocked = writer.is_alive()
    finally:
        release.set()
        reader.join(5)
    assert not blocked, "append() is waiting for the reader: the lock is held during the pandas work"
    assert appended == [True]
    assert_stamps(result[0], [1, 2, 3])  # the window is the snapshot from before that row


@pytest.fixture
def fast_thread_switching():
    """Make Python switch threads after almost every few instructions: that makes races much more likely."""
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)
    yield
    sys.setswitchinterval(previous)


def test_one_writer_and_several_readers_never_see_an_inconsistent_frame(fast_thread_switching):
    step, total, capacity, readers = 0.05, 10_000, 400, 4
    reset_at = {2_500, 5_000, 7_500}  # the system clock jumps back 500 s here
    names = ["twice", "minus", "thousand", "flag"]
    cache = SampleCache(names, retention_s=FOREVER, max_samples=capacity, flags=["flag"], clock=Clock(T0))

    gate = threading.Barrier(readers + 1)
    writer_done = threading.Event()
    failures: list[BaseException] = []
    newest = [T0]
    reads = [0] * readers
    nonempty = [0] * readers
    expected = {"invalid": 0, "out_of_order": 0}

    def let_readers_catch_up() -> None:
        """Wait until every reader has done a few more rounds: this guarantees concurrent use
        while data is still arriving, even if the machine is busy."""
        goal = sum(reads) + 3 * readers
        deadline = time.monotonic() + 10
        while sum(reads) < goal and not failures and time.monotonic() < deadline:
            time.sleep(0.001)

    def write():
        try:
            gate.wait(10)
            base = T0
            for i in range(total):
                if i in reset_at:
                    base -= 500.0
                if i in (1_250, 6_250):  # in the first part and in the third part
                    let_readers_catch_up()
                t = base + i * step
                offset = t - T0
                # All columns follow linearly from the time, so every row (also after averaging) can be checked.
                values = {"twice": 2 * offset, "minus": -offset, "thousand": 1_000 * offset, "flag": float(i % 50 == 0)}
                assert cache.append(t, values) is True
                newest[0] = t
                if i % 61 == 0:
                    assert cache.append(t - 1.0, values) is False
                    expected["out_of_order"] += 1
                if i % 97 == 0:
                    assert cache.append(t + 1.0, {**values, "minus": math.nan}) is False
                    expected["invalid"] += 1
                if i % 25 == 0:
                    time.sleep(0)  # give the readers ample opportunity between the writes
        except BaseException as err:  # noqa: BLE001 - pass on to the test thread
            failures.append(err)
        finally:
            writer_done.set()

    def check(df: pd.DataFrame, start: float, end: float, limit: int) -> None:
        assert list(df.columns) == names and (df.dtypes == np.float64).all()
        assert isinstance(df.index, pd.DatetimeIndex) and str(df.index.tz) == "UTC"
        assert len(df) <= limit
        if df.empty:
            return
        stamps = epoch(df)
        assert (np.diff(stamps) > 0).all(), "times not strictly increasing"
        assert stamps[0] >= start - 1e-6 and stamps[-1] <= end + 1e-6, "row outside the requested window"
        offset = stamps - T0
        # every row is consistent with itself: a row from two different writes is caught here
        assert np.abs(df["twice"].to_numpy() - 2 * offset).max() < 2e-6
        assert np.abs(df["minus"].to_numpy() + offset).max() < 2e-6
        assert np.abs(df["thousand"].to_numpy() - 1_000 * offset).max() < 2e-3
        assert df["flag"].isin([0.0, 1.0]).all(), "a flag is no longer 0/1"

    def read(number: int):
        rng = random.Random(number)
        try:
            gate.wait(10)
            last_round = False
            while not last_round:
                last_round = writer_done.is_set()  # check first, then read: one final round after the end
                limit = rng.choice([1, 5, 50, 1_000])
                if last_round or rng.random() < 0.2:
                    start, end = -math.inf, math.inf  # the last round reads everything, so it is never empty
                else:
                    start = newest[0] - rng.uniform(0, 40)
                    end = start + rng.uniform(-1, 40)  # sometimes the end lies before the start
                df = cache.window(start, end, limit)
                check(df, start, end, limit)
                reads[number] += 1
                nonempty[number] += not df.empty
        except BaseException as err:  # noqa: BLE001 - pass on to the test thread
            failures.append(err)

    threads = [threading.Thread(target=write, name="writer", daemon=True)]
    threads += [threading.Thread(target=read, args=(n,), name=f"reader-{n}", daemon=True) for n in range(readers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)

    assert not [t.name for t in threads if t.is_alive()], "a thread hangs: deadlock?"
    assert not failures, failures[:3]
    assert all(count >= 1 for count in nonempty)  # every reader has seen at least the last, filled round

    stats = cache.stats()
    assert stats.appended == total and stats.clock_resets == len(reset_at)
    assert stats.dropped_out_of_order == expected["out_of_order"] and stats.dropped_invalid == expected["invalid"]
    assert stats.rows == capacity  # the last part (2,500 rows) is much larger than the ring
    assert stats.overwritten == 4 * (2_500 - capacity)  # four parts between the clock resets


# ---- together with the existing analysis ---------------------------------------------------------------


def test_frames_feed_straight_into_the_existing_analysis():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    source = MockSource().read(start, start + timedelta(minutes=2), 40_000)
    cache = SampleCache(
        SIGNALS, retention_s=600, max_samples=5_000, flags=["system_mode"], clock=Clock(epoch(source)[-1])
    )
    for stamp, values in zip(epoch(source).tolist(), source.to_dict("records"), strict=True):
        assert cache.append(stamp, values)

    df = cache.window(-math.inf, math.inf, 40_000)

    assert list(df.columns) == SIGNALS and len(df) == len(source)
    np.testing.assert_allclose(epoch(df), epoch(source), rtol=0, atol=1e-6)
    np.testing.assert_allclose(df.to_numpy(), source[SIGNALS].to_numpy())
    derived = analysis.add_derived(df, CylinderGeometry())
    overview = analysis.overview(derived)
    assert overview["samples"] == len(df)
    assert overview["dominant_frequency_hz"] == pytest.approx(TOWER_FREQ_HZ, abs=0.03)
    assert len(analysis.decimate(derived, 200)) <= 202
    thin = cache.window(-math.inf, math.inf, 100)  # a downsampled window can go straight in too
    assert 0 < len(thin) <= 100 and not analysis.add_derived(thin, CylinderGeometry()).isna().any().any()
