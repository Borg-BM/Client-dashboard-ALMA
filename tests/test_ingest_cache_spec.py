"""Independent spec tests for alma.ingest.cache (SampleCache and CacheStats).

This only tests the public API from the spec, not the implementation. Timestamps lie on a grid of
1/8 s (or 1/64 s) around T0: distances, thresholds and window bounds can then be written exactly in float64, so
'exactly on the bound' really is exactly on the bound, regardless of the resolution of the pandas index.
"""

import dataclasses
import gc
import inspect
import itertools
import json
import math
import os
import random
import re
import sys
import threading
import time
import tracemalloc
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from alma.ingest.cache import CacheStats, SampleCache

T0 = 1_700_000_000.0
STEP = 0.25
SIGS = ("b_analog", "a_analog", "flag")  # deliberately not alphabetical: column order is the constructor order
FLAGS = ("flag",)
EVERYTHING = (-1e18, 1e18)
UNLIMITED = 10**9
TIME_ATOL = 1e-6  # raw rows on the 1/8 s grid
TIME_ATOL_MEAN = 1e-3  # mean times: the resolution of the pandas index is not in the spec
NAN, INF = float("nan"), float("inf")


class FakeClock:
    """Clock that the test advances itself."""

    def __init__(self, now: float = T0 + 100_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def stamp(i: int) -> float:
    return T0 + i * STEP


def payload(i: int) -> dict[str, float]:
    """Values of row i; 'a_analog' carries the row number so that a frame identifies itself."""
    return {"b_analog": 10.0 * i + 1.0, "a_analog": float(i), "flag": 1.0 if i % 3 == 0 else 0.0}


def make(capacity=100, *, clock=None, retention_s=1e6, signals=SIGS, flags=FLAGS, step_back=5.0) -> SampleCache:
    return SampleCache(
        signals,
        retention_s=retention_s,
        max_samples=capacity,
        flags=flags,
        clock=clock or FakeClock(),
        clock_step_back_s=step_back,
    )


def fill(cache: SampleCache, indices) -> None:
    for i in indices:
        assert cache.append(stamp(i), payload(i)) is True, i


def everything(cache: SampleCache, max_points: int = UNLIMITED) -> pd.DataFrame:
    return cache.window(*EVERYTHING, max_points)


def epoch(index: pd.DatetimeIndex) -> np.ndarray:
    """Index as epoch seconds, regardless of the resolution (s, ms, us or ns) of the index."""
    return ((index - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def ids(df: pd.DataFrame) -> list[int]:
    return df["a_analog"].astype(int).tolist()


def assert_utc_index(index) -> None:
    assert isinstance(index, pd.DatetimeIndex)
    assert isinstance(index.dtype, pd.DatetimeTZDtype)
    assert str(index.tz) in ("UTC", "Etc/UTC")


def assert_typed(df: pd.DataFrame, signals=SIGS) -> None:
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == list(signals)
    assert all(dtype == np.float64 for dtype in df.dtypes)
    assert_utc_index(df.index)


def assert_empty(df: pd.DataFrame, signals=SIGS) -> None:
    assert_typed(df, signals)
    assert len(df) == 0 and df.empty


def assert_rows(df: pd.DataFrame, indices, signals=SIGS) -> None:
    """The frame contains exactly the rows payload(i)/stamp(i) for i in indices, in that order."""
    indices = list(indices)
    assert_typed(df, signals)
    assert len(df) == len(indices)
    np.testing.assert_allclose(epoch(df.index), [stamp(i) for i in indices], rtol=0, atol=TIME_ATOL)
    expected = np.array([[payload(i)[s] for s in signals] for i in indices], dtype=float)
    np.testing.assert_array_equal(df.to_numpy(), expected.reshape(len(indices), len(signals)))


def check_stats(cache: SampleCache, **expected) -> None:
    stats = cache.stats()
    for name, want in expected.items():
        got = getattr(stats, name)
        if want is None:
            assert got is None, (name, got)
        elif isinstance(want, float):
            assert got == pytest.approx(want, rel=0, abs=TIME_ATOL), (name, got, want)
        else:
            assert got == want, (name, got, want)


# English function words and stems; a Dutch message contains none of them
ENGLISH_WORDS = frozenset(
    """must not least greater smaller higher lower finite infinite unique duplicate duplicates empty contains
    contain missing unknown invalid signal signals flag flags subset only are is be got integer number names
    string callable required needed than at the in of a an""".split()
)


def assert_english(message: str) -> None:
    words = set(re.findall(r"[a-z]+", message.lower()))
    assert message.strip(), "empty error message"
    assert words & ENGLISH_WORDS, f"error message does not look English: {message!r}"


# ---- constructor -----------------------------------------------------------------------------------

INVALID_ARGUMENTS = {
    "empty signals list": {"signals": []},
    "empty signals tuple": {"signals": ()},
    "duplicate signals": {"signals": ["a", "b", "a"]},
    "retention zero": {"retention_s": 0.0},
    "retention negative": {"retention_s": -1.0},
    "retention nan": {"retention_s": NAN},
    "retention inf": {"retention_s": INF},
    "retention minus inf": {"retention_s": -INF},
    "max_samples one": {"max_samples": 1},
    "max_samples zero": {"max_samples": 0},
    "max_samples negative": {"max_samples": -5},
    "step back zero": {"clock_step_back_s": 0.0},
    "step back negative": {"clock_step_back_s": -1.0},
    "flag is not a signal": {"flags": ["x"]},
    "one flag is not a signal": {"flags": ["a", "x"]},
    "flags as set, not a subset": {"flags": {"zzz"}},
}


def build(**overrides) -> SampleCache:
    args = {"signals": ["a", "b"], "retention_s": 60.0, "max_samples": 10} | overrides
    signals = args.pop("signals")
    return SampleCache(signals, **args)


@pytest.mark.parametrize("case", list(INVALID_ARGUMENTS))
def test_constructor_rejects_invalid_arguments_with_an_english_value_error(case):
    with pytest.raises(ValueError) as caught:
        build(**INVALID_ARGUMENTS[case])
    assert_english(str(caught.value))


def test_constructor_accepts_the_boundary_values():
    assert len(build(max_samples=2)) == 0
    assert len(build(retention_s=1e-9)) == 0
    assert len(build(retention_s=60)) == 0  # an int is a number too
    assert len(build(clock_step_back_s=1e-9)) == 0
    assert len(build(signals=("only",))) == 0
    assert build(max_samples=2).stats().capacity == 2


@pytest.mark.parametrize("flags", [(), [], set(), frozenset(), ("b",), ["a", "b"], {"a"}, frozenset({"a", "b"})])
def test_constructor_accepts_any_collection_of_flags(flags):
    assert len(build(flags=flags)) == 0


def test_constructor_signature_matches_the_spec():
    params = inspect.signature(SampleCache).parameters
    assert next(iter(params)) == "signals"
    for name in ("retention_s", "max_samples", "flags", "clock", "clock_step_back_s"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name
    assert params["retention_s"].default is inspect.Parameter.empty
    assert params["max_samples"].default is inspect.Parameter.empty
    assert len(params["flags"].default) == 0
    assert params["clock"].default is time.time
    assert params["clock_step_back_s"].default == 5.0
    with pytest.raises(TypeError):
        SampleCache(["a"], 60.0, 10)  # positional is not allowed
    with pytest.raises(TypeError):
        SampleCache(["a"], max_samples=10)
    with pytest.raises(TypeError):
        SampleCache(["a"], retention_s=60.0)


def test_constructor_takes_a_copy_of_signals_and_flags():
    signals, flags = ["b_analog", "a_analog", "flag"], {"flag"}
    cache = SampleCache(signals, retention_s=1e6, max_samples=10, flags=flags, clock=FakeClock())
    signals.reverse()
    signals.append("extra")
    flags.clear()
    flags.add("b_analog")
    fill(cache, range(6))
    assert list(everything(cache).columns) == list(SIGS)
    # 'flag' stayed a flag (max) and 'b_analog' did not become one (mean): rows 0..5, k = 3
    small = everything(cache, 2)
    assert small["flag"].tolist() == [1.0, 1.0]
    assert small["b_analog"].tolist() == [pytest.approx((1 + 11 + 21) / 3), pytest.approx((31 + 41 + 51) / 3)]


def test_signal_names_may_look_like_pandas_attributes():
    names = ["index", "values", "T", "drill pressure (bar)", "count"]
    cache = SampleCache(names, retention_s=1e6, max_samples=8, flags=["T"], clock=FakeClock())
    for i in range(3):
        assert cache.append(stamp(i), {n: float(i * 10 + j) for j, n in enumerate(names)}) is True
    df = everything(cache)
    assert list(df.columns) == names
    assert df["values"].tolist() == [1.0, 11.0, 21.0]
    assert df["T"].tolist() == [2.0, 12.0, 22.0]
    assert df["drill pressure (bar)"].tolist() == [3.0, 13.0, 23.0]


@pytest.mark.parametrize("count", [1, 2, 50])
def test_cache_works_for_one_two_and_many_signals(count):
    names = [f"s{i:02d}" for i in range(count)][::-1]  # reversed: constructor order beats alphabet
    cache = SampleCache(names, retention_s=1e6, max_samples=4, flags=names[:1], clock=FakeClock())
    rng = np.random.default_rng(count)
    history = rng.normal(size=(9, count))
    for i, row in enumerate(history):
        assert cache.append(stamp(i), dict(zip(names, row))) is True
    df = everything(cache)
    assert list(df.columns) == names and df.shape == (4, count)
    np.testing.assert_array_equal(df.to_numpy(), history[-4:])
    two = everything(cache, 2)  # k = 2: column 0 is a flag (max), the rest mean
    expected_mean = history[-4:].reshape(2, 2, count).mean(axis=1)
    expected_max = history[-4:].reshape(2, 2, count).max(axis=1)
    np.testing.assert_allclose(two.to_numpy()[:, 1:], expected_mean[:, 1:], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(two.to_numpy()[:, 0], expected_max[:, 0])


# ---- CacheStats ------------------------------------------------------------------------------------

STAT_FIELDS = [
    "rows",
    "capacity",
    "oldest",
    "newest",
    "appended",
    "dropped_out_of_order",
    "dropped_invalid",
    "overwritten",
    "clock_resets",
]


def test_cache_stats_is_a_frozen_dataclass_with_the_documented_fields_in_order():
    assert dataclasses.is_dataclass(CacheStats)
    assert [f.name for f in dataclasses.fields(CacheStats)] == STAT_FIELDS
    stats = CacheStats(1, 2, 3.0, 4.0, 5, 6, 7, 8, 9)  # order of the spec
    assert [getattr(stats, name) for name in STAT_FIELDS] == [1, 2, 3.0, 4.0, 5, 6, 7, 8, 9]
    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.rows = 99
    assert stats == CacheStats(1, 2, 3.0, 4.0, 5, 6, 7, 8, 9)
    assert hash(stats) == hash(CacheStats(1, 2, 3.0, 4.0, 5, 6, 7, 8, 9))
    assert CacheStats(0, 5, None, None, 0, 0, 0, 0, 0).oldest is None


def test_stats_of_an_empty_cache():
    stats = make(capacity=7).stats()
    assert isinstance(stats, CacheStats)
    assert (stats.rows, stats.capacity, stats.oldest, stats.newest) == (0, 7, None, None)
    assert (stats.appended, stats.dropped_out_of_order, stats.dropped_invalid) == (0, 0, 0)
    assert (stats.overwritten, stats.clock_resets) == (0, 0)


def test_stats_fields_have_plain_python_types():
    cache = make(capacity=3)
    fill(cache, range(5))
    cache.append(stamp(1), payload(1))
    cache.append(NAN, payload(1))
    stats = cache.stats()
    for name in STAT_FIELDS:
        if name in ("oldest", "newest"):
            assert isinstance(getattr(stats, name), float), name
        else:
            value = getattr(stats, name)
            assert isinstance(value, int) and not isinstance(value, bool), name
    assert isinstance(len(cache), int)
    json.dumps(dataclasses.asdict(stats))  # the health API must be able to serialize this


def test_every_counter_moves_exactly_as_specified():
    cache = make(capacity=4, step_back=2.0)
    fill(cache, range(6))
    check_stats(cache, rows=4, capacity=4, appended=6, overwritten=2, oldest=stamp(2), newest=stamp(5))

    assert cache.append(NAN, payload(6)) is False
    assert cache.append(stamp(6), payload(6) | {"flag": NAN}) is False
    check_stats(cache, dropped_invalid=2, dropped_out_of_order=0, rows=4, appended=6)

    assert cache.append(stamp(5), payload(5)) is False  # equal
    assert cache.append(stamp(4), payload(4)) is False  # older
    assert cache.append(stamp(-3), payload(-3)) is False  # newest - ts == 2.0: exactly the threshold, no reset
    check_stats(cache, dropped_out_of_order=3, clock_resets=0, rows=4, appended=6, overwritten=2)

    assert cache.append(stamp(-4), payload(-4)) is True  # newest - ts == 2.25 > 2.0: reset
    check_stats(
        cache,
        clock_resets=1,
        rows=1,
        appended=7,
        overwritten=2,
        dropped_out_of_order=3,
        dropped_invalid=2,
        oldest=stamp(-4),
        newest=stamp(-4),
    )

    fill(cache, range(-3, 0))  # fills the ring again: -4..-1
    check_stats(cache, rows=4, appended=10, overwritten=2, oldest=stamp(-4), newest=stamp(-1))
    fill(cache, [0])  # now -4 is overwritten
    check_stats(cache, rows=4, appended=11, overwritten=3, oldest=stamp(-3), newest=stamp(0), clock_resets=1)
    assert len(cache) == 4


# ---- append: invalid and special input --------------------------------------------------------


def test_append_stores_rows_and_returns_true():
    cache = make()
    assert cache.append(stamp(0), payload(0)) is True
    assert len(cache) == 1
    assert_rows(everything(cache), [0])


@pytest.mark.parametrize("missing", SIGS)
def test_row_with_a_missing_signal_is_dropped_as_invalid(missing):
    cache = make()
    fill(cache, range(3))
    bad = {k: v for k, v in payload(3).items() if k != missing}
    assert cache.append(stamp(3), bad) is False
    check_stats(cache, rows=3, appended=3, dropped_invalid=1, dropped_out_of_order=0, clock_resets=0)
    assert_rows(everything(cache), range(3))
    assert cache.append(stamp(3), payload(3)) is True  # a rejected row does not 'consume' the timestamp


def test_empty_values_are_dropped_as_invalid():
    cache = make()
    assert cache.append(stamp(0), {}) is False
    check_stats(cache, rows=0, appended=0, dropped_invalid=1)


@pytest.mark.parametrize("bad", [NAN, INF, -INF])
@pytest.mark.parametrize("signal", SIGS)
def test_row_with_a_non_finite_value_is_dropped_as_invalid(signal, bad):
    cache = make()
    fill(cache, range(3))
    assert cache.append(stamp(3), payload(3) | {signal: bad}) is False
    check_stats(cache, rows=3, appended=3, dropped_invalid=1, dropped_out_of_order=0, clock_resets=0)
    assert_rows(everything(cache), range(3))
    assert cache.append(stamp(3), payload(3)) is True


@pytest.mark.parametrize("prefill", [0, 3])
@pytest.mark.parametrize("bad", [NAN, INF, -INF])
def test_non_finite_timestamp_is_dropped_as_invalid(bad, prefill):
    cache = make()
    fill(cache, range(prefill))
    assert cache.append(bad, payload(9)) is False
    # -inf is 'older than everything': so it must not cause a reset; +inf must not become the newest
    check_stats(cache, rows=prefill, appended=prefill, dropped_invalid=1, dropped_out_of_order=0, clock_resets=0)
    assert_rows(everything(cache), range(prefill))
    assert cache.append(stamp(prefill), payload(prefill)) is True


def test_extra_keys_are_ignored_even_when_they_are_not_finite():
    cache = make()
    row = payload(0) | {"junk": NAN, "other": INF, "text": "x", "nothing": None}
    assert cache.append(stamp(0), row) is True
    assert_rows(everything(cache), [0])
    check_stats(cache, dropped_invalid=0, appended=1)


def test_a_row_with_several_problems_is_counted_once():
    cache = make()
    assert cache.append(NAN, {"b_analog": NAN, "a_analog": INF}) is False  # 'flag' is missing too
    check_stats(cache, dropped_invalid=1, rows=0, appended=0, dropped_out_of_order=0)


def test_values_are_matched_by_name_not_by_order():
    cache = make()
    shuffled = dict(reversed(list(payload(7).items())))
    assert cache.append(stamp(7), shuffled) is True
    assert_rows(everything(cache), [7])


class OtherMapping(Mapping):
    """A Mapping that is not a dict."""

    def __init__(self, data):
        self._data = dict(data)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


def test_values_may_be_any_mapping_and_any_real_number_type():
    cache = make()
    assert cache.append(stamp(0), MappingProxyType(payload(0))) is True
    assert cache.append(stamp(1), OtherMapping(payload(1))) is True
    assert cache.append(stamp(2), {"b_analog": np.float32(1.5), "a_analog": np.float64(2.0), "flag": np.int64(1)}) is True
    assert cache.append(stamp(3), {"b_analog": 3, "a_analog": 4, "flag": 0}) is True
    df = everything(cache)
    assert_typed(df)
    np.testing.assert_array_equal(
        df.to_numpy(),
        [[1.0, 0.0, 1.0], [11.0, 1.0, 0.0], [1.5, 2.0, 1.0], [3.0, 4.0, 0.0]],
    )


def test_timestamps_may_be_ints_and_numpy_numbers():
    cache = make()
    base = int(T0)
    assert cache.append(base, payload(0)) is True
    assert cache.append(np.float64(base + 1.5), payload(1)) is True
    assert cache.append(np.int64(base + 3), payload(2)) is True
    np.testing.assert_allclose(epoch(everything(cache).index), [base, base + 1.5, base + 3], rtol=0, atol=TIME_ATOL)


@pytest.mark.parametrize(
    "spoil", ["nan_in_last_signal", "nan_in_first_signal", "missing_last_signal", "nan_timestamp", "inf_timestamp"]
)
def test_a_dropped_row_never_alters_stored_rows_even_when_the_ring_is_full(spoil):
    cache = make(capacity=3)
    fill(cache, range(5))  # full and overwritten twice: rows 2, 3, 4
    before = everything(cache)
    overwritten = cache.stats().overwritten
    bad, ts = payload(5), stamp(5)
    if spoil == "nan_in_last_signal":
        bad["flag"] = NAN
    elif spoil == "nan_in_first_signal":
        bad["b_analog"] = NAN
    elif spoil == "missing_last_signal":
        del bad["flag"]
    elif spoil == "nan_timestamp":
        ts = NAN
    else:
        ts = INF
    assert cache.append(ts, bad) is False
    assert cache.stats().overwritten == overwritten  # nothing was overwritten
    pd.testing.assert_frame_equal(everything(cache), before)
    assert ids(everything(cache)) == [2, 3, 4]
    assert cache.append(stamp(5), payload(5)) is True


@pytest.mark.parametrize(
    "timestamp, values",
    [
        (None, payload(9)),
        ("no time", payload(9)),
        (stamp(9), None),
        (stamp(9), {"b_analog": None, "a_analog": 1.0, "flag": 0.0}),
        (stamp(9), {"b_analog": 1.0, "a_analog": "abc", "flag": 0.0}),
        (10**400, payload(9)),
        (stamp(9), {"b_analog": 1.0, "a_analog": 2.0, "flag": 10**400}),
    ],
)
def test_bad_data_of_any_type_is_dropped_as_invalid_and_never_raises(timestamp, values):
    # Spec: append() 'NEVER raises for bad data'. Whatever kind of junk arrives, it is False and invalid.
    cache = make(capacity=3)
    fill(cache, range(5))
    before, stats_before = everything(cache), cache.stats()
    assert cache.append(timestamp, values) is False
    pd.testing.assert_frame_equal(everything(cache), before)
    stats_after = cache.stats()
    assert stats_after.dropped_invalid == stats_before.dropped_invalid + 1
    assert (stats_after.rows, stats_after.appended, stats_after.overwritten, stats_after.clock_resets) == (
        stats_before.rows,
        stats_before.appended,
        stats_before.overwritten,
        stats_before.clock_resets,
    )
    assert stats_after.dropped_out_of_order == stats_before.dropped_out_of_order
    assert cache.append(stamp(5), payload(5)) is True


@pytest.mark.parametrize("huge", [1e10, -1e10, 1e18, 1e300, sys.float_info.max, -sys.float_info.max])
def test_a_finite_timestamp_far_outside_the_datetime_range_never_breaks_window(huge):
    # The spec only calls non-finite times invalid. A finite time beyond datetime64[ns] (year 2262) may
    # be stored or rejected, but must never make append() raise or a later window() fail.
    cache = make(capacity=4)
    fill(cache, range(3))
    assert cache.append(huge, payload(9)) in (True, False)
    for max_points in (UNLIMITED, 1):
        df = everything(cache, max_points)
        assert_typed(df)
        assert df.index.is_unique and df.index.is_monotonic_increasing
    assert cache.append(stamp(50), payload(50)) in (True, False)
    assert_typed(everything(cache))


def test_append_does_not_depend_on_the_clock():
    clock = FakeClock(T0 - 10_000.0)  # clock far before the data
    cache = make(clock=clock, retention_s=60.0)
    fill(cache, range(5))
    clock.now = T0 + 10_000.0  # and now far after it: all rows are older than the retention
    assert cache.append(stamp(5), payload(5)) is True
    check_stats(cache, rows=6, appended=6, oldest=stamp(0), newest=stamp(5))
    assert_empty(everything(cache))


# ---- ring buffer ------------------------------------------------------------------------------------


@pytest.mark.parametrize("capacity", [2, 3, 5])
def test_ring_keeps_the_newest_rows_in_order_after_every_append(capacity):
    cache = make(capacity=capacity)
    for n in range(1, 4 * capacity + 4):
        assert cache.append(stamp(n), payload(n)) is True
        expected = list(range(max(1, n - capacity + 1), n + 1))
        assert_rows(everything(cache), expected)
        assert len(cache) == len(expected)
        check_stats(
            cache,
            rows=len(expected),
            capacity=capacity,
            oldest=stamp(expected[0]),
            newest=stamp(n),
            appended=n,
            overwritten=n - len(expected),
            dropped_out_of_order=0,
            dropped_invalid=0,
            clock_resets=0,
        )
        middle = expected[len(expected) // 2]  # a window that starts in the middle of the ring
        assert_rows(cache.window(stamp(middle), stamp(n), UNLIMITED), range(middle, n + 1))
        assert_rows(cache.window(stamp(n), stamp(n), UNLIMITED), [n])


@pytest.mark.parametrize("capacity", [2, 3, 5])
def test_ring_with_random_values_matches_a_sliding_window_reference(capacity):
    rng = np.random.default_rng(capacity)
    cache = make(capacity=capacity)
    history = []
    for n in range(40):
        values = rng.normal(size=3) * 100
        assert cache.append(stamp(n), dict(zip(SIGS, values))) is True
        history.append(values)
        df = everything(cache)
        np.testing.assert_array_equal(df.to_numpy(), np.array(history[-capacity:]))
        first = max(0, n - capacity + 1)
        np.testing.assert_allclose(epoch(df.index), [stamp(i) for i in range(first, n + 1)], rtol=0, atol=TIME_ATOL)


# ---- window: bounds, retention, arguments -----------------------------------------------------


def test_window_bounds_are_inclusive_on_both_ends():
    cache = make()
    fill(cache, range(10))
    cases = [
        ((stamp(3), stamp(6)), [3, 4, 5, 6]),
        ((stamp(3), stamp(3)), [3]),
        ((stamp(0), stamp(9)), list(range(10))),
        ((stamp(0) - 1, stamp(9) + 1), list(range(10))),
        ((stamp(0) + 0.125, stamp(9)), list(range(1, 10))),
        ((stamp(0), stamp(9) - 0.125), list(range(9))),
        ((stamp(3) + 0.125, stamp(4) - 0.125), []),
        ((stamp(3) + 0.125, stamp(5) - 0.125), [4]),
        ((stamp(9) + 0.125, stamp(20)), []),
        ((stamp(-5), stamp(0) - 0.125), []),
    ]
    for (start, end), wanted in cases:
        assert_rows(cache.window(start, end, UNLIMITED), wanted)


def test_window_bounds_are_exact_to_the_last_float_bit():
    cache = make()
    fill(cache, range(10))
    t = stamp(4)
    below, above = float(np.nextafter(t, -np.inf)), float(np.nextafter(t, np.inf))
    assert ids(cache.window(below, t, UNLIMITED)) == [4]
    assert ids(cache.window(t, above, UNLIMITED)) == [4]
    assert ids(cache.window(above, 1e18, UNLIMITED)) == [5, 6, 7, 8, 9]
    assert ids(cache.window(-1e18, below, UNLIMITED)) == [0, 1, 2, 3]
    assert ids(cache.window(below, below, UNLIMITED)) == []


def test_window_accepts_infinite_bounds():
    cache = make()
    fill(cache, range(6))
    assert ids(cache.window(-math.inf, math.inf, UNLIMITED)) == list(range(6))
    assert ids(cache.window(stamp(2), math.inf, UNLIMITED)) == [2, 3, 4, 5]
    assert ids(cache.window(-math.inf, stamp(2), UNLIMITED)) == [0, 1, 2]


def test_window_with_start_after_end_is_empty_even_when_rows_lie_between():
    cache = make()
    fill(cache, range(10))
    for start, end in [(stamp(6), stamp(3)), (stamp(5), stamp(5) - 0.125), (1e18, -1e18), (stamp(9), stamp(0))]:
        assert_empty(cache.window(start, end, UNLIMITED))
        assert_empty(cache.window(start, end, 1))


@pytest.mark.parametrize("prefill", [0, 5])
@pytest.mark.parametrize("bad", [0, -1, -100])
def test_window_rejects_max_points_below_one(bad, prefill):
    cache = make()
    fill(cache, range(prefill))
    with pytest.raises(ValueError):
        cache.window(*EVERYTHING, bad)
    assert len(cache) == prefill
    assert len(cache.window(*EVERYTHING, 1)) == min(prefill, 1)  # the cache remains usable


@pytest.mark.parametrize("huge", [10**9, 2**31, sys.maxsize])
def test_window_with_a_huge_max_points_returns_the_rows_unchanged(huge):
    cache = make()
    fill(cache, range(8))
    assert_rows(cache.window(*EVERYTHING, huge), range(8))


def test_windows_are_repeatable_and_have_no_side_effects():
    cache = make(capacity=6)
    fill(cache, range(9))
    stats = cache.stats()
    for max_points in (UNLIMITED, 2):
        first = everything(cache, max_points)
        pd.testing.assert_frame_equal(everything(cache, max_points), first)
    assert cache.stats() == stats
    assert len(cache) == 6


def test_retention_cutoff_is_inclusive_and_follows_the_injected_clock():
    clock = FakeClock(T0 + 100.0)
    cache = make(capacity=1000, clock=clock, retention_s=50.0)  # bound = T0 + 50 = row 200
    fill(cache, range(401))
    assert ids(everything(cache)) == list(range(200, 401))  # exactly on the bound counts
    clock.now = T0 + 100.25  # bound moves up one row
    assert ids(everything(cache)) == list(range(201, 401))
    clock.now = T0 + 99.75
    assert ids(everything(cache)) == list(range(199, 401))  # the row was never discarded, so it comes back
    clock.now = T0 + 500.0
    assert_empty(everything(cache))
    assert len(cache) == 401
    clock.now = T0 + 100.0
    assert ids(everything(cache)) == list(range(200, 401))


def test_retention_cutoff_is_exact_to_the_last_float_bit():
    clock = FakeClock(T0 + 100.0)
    cache = make(clock=clock, retention_s=50.0)
    cutoff = T0 + 50.0
    below, above = float(np.nextafter(cutoff, -np.inf)), float(np.nextafter(cutoff, np.inf))
    for i, t in enumerate([below, cutoff, above]):
        assert cache.append(t, payload(i)) is True
    assert ids(everything(cache)) == [1, 2]


def test_retention_and_start_combine_to_the_stricter_bound():
    clock = FakeClock(T0 + 100.0)
    cache = make(capacity=1000, clock=clock, retention_s=50.0)  # bound = row 200
    fill(cache, range(401))
    assert ids(cache.window(stamp(240), stamp(260), UNLIMITED)) == list(range(240, 261))  # start stricter
    assert ids(cache.window(stamp(10), stamp(210), UNLIMITED)) == list(range(200, 211))  # retention stricter
    assert_empty(cache.window(stamp(10), stamp(199), UNLIMITED))  # end before the bound
    assert ids(cache.window(stamp(200), stamp(200), UNLIMITED)) == [200]


def test_rows_newer_than_the_clock_are_returned():
    cache = make(clock=FakeClock(T0 - 1_000.0), retention_s=60.0)  # only a lower bound, no upper bound
    fill(cache, range(10))
    assert ids(everything(cache)) == list(range(10))


def test_default_clock_is_the_wall_clock():
    cache = SampleCache(SIGS, retention_s=60.0, max_samples=10, flags=FLAGS)
    now = time.time()
    assert cache.append(now - 120.0, payload(0)) is True  # older than the retention: stored after all...
    assert cache.append(now - 1.0, payload(1)) is True
    assert len(cache) == 2
    assert ids(cache.window(0.0, now + 3_600.0, 100)) == [1]  # ...but not returned


def test_len_and_stats_count_rows_regardless_of_retention():
    clock = FakeClock(T0 + 100.0)
    cache = make(clock=clock, retention_s=10.0)
    fill(cache, range(40))
    clock.now = T0 + 10_000.0
    assert_empty(everything(cache))
    assert len(cache) == 40
    check_stats(cache, rows=40, oldest=stamp(0), newest=stamp(39), appended=40)


# ---- thinning out -------------------------------------------------------------------------------------

HAND_B = [10.0, 0.0, 20.0, 5.0, 5.0, 5.0, -4.0]
HAND_A = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
HAND_FLAG = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
HAND_TIMES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
HAND_ROWS = [[b, a, f] for b, a, f in zip(HAND_B, HAND_A, HAND_FLAG)]
# max_points -> (times relative to T0, rows [b, a, flag]); worked out by hand from the 7 rows above
HAND_EXPECTED = {
    1: ([0.75], [[41 / 7, 4.0, 1.0]]),  # k = 7
    2: ([0.375, 1.25], [[8.75, 2.5, 1.0], [2.0, 6.0, 0.0]]),  # k = 4: buckets of 4 and 3
    3: ([0.25, 1.0, 1.5], [[10.0, 2.0, 1.0], [5.0, 5.0, 0.0], [-4.0, 7.0, 0.0]]),  # k = 3: 3, 3, 1
    4: ([0.125, 0.625, 1.125, 1.5], [[5.0, 1.5, 0.0], [12.5, 3.5, 1.0], [5.0, 5.5, 0.0], [-4.0, 7.0, 0.0]]),  # k = 2
    5: ([0.125, 0.625, 1.125, 1.5], [[5.0, 1.5, 0.0], [12.5, 3.5, 1.0], [5.0, 5.5, 0.0], [-4.0, 7.0, 0.0]]),
    6: ([0.125, 0.625, 1.125, 1.5], [[5.0, 1.5, 0.0], [12.5, 3.5, 1.0], [5.0, 5.5, 0.0], [-4.0, 7.0, 0.0]]),
    7: (HAND_TIMES, HAND_ROWS),  # as many rows as max_points: unchanged
    8: (HAND_TIMES, HAND_ROWS),
}


@pytest.mark.parametrize("max_points", sorted(HAND_EXPECTED))
def test_downsampling_hand_checked_example(max_points):
    cache = make(capacity=20)
    for offset, row in zip(HAND_TIMES, HAND_ROWS):
        assert cache.append(T0 + offset, dict(zip(SIGS, row))) is True
    df = cache.window(*EVERYTHING, max_points)
    times, rows = HAND_EXPECTED[max_points]
    assert_typed(df)
    np.testing.assert_allclose(epoch(df.index), T0 + np.array(times), rtol=0, atol=TIME_ATOL_MEAN)
    np.testing.assert_allclose(df.to_numpy(), rows, rtol=1e-12, atol=1e-12)


def bucket_reference(times, values, flag_columns, max_points):
    """Independent implementation of the spec: consecutive buckets of k = ceil(n / max_points) rows."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)  # (n, columns); also empty as (0, columns)
    n = len(times)
    if n <= max_points:
        return times, values
    k = -(-n // max_points)
    out_t, out_v = [], []
    for lo in range(0, n, k):
        hi = min(lo + k, n)
        out_t.append(T0 + float(np.mean(times[lo:hi] - T0)))
        out_v.append(
            [float(np.max(values[lo:hi, j])) if j in flag_columns else float(np.mean(values[lo:hi, j])) for j in range(values.shape[1])]
        )
    return np.array(out_t), np.array(out_v)


def random_rows(n, seed):
    """Irregular times on the 1/8 s grid; columns in SIGS order (b_analog, a_analog, flag)."""
    rng = np.random.default_rng(seed)
    times = T0 + np.cumsum(rng.choice([1, 2, 3, 5, 8], size=n) * 0.125)
    values = np.column_stack([rng.normal(0, 100, n), rng.normal(0, 100, n), rng.choice([0.0, 1.0, 2.5, -1.0], n)])
    return times, values


def loaded_cache(times, values, flags=FLAGS) -> SampleCache:
    cache = make(capacity=len(times) + 3, flags=flags)
    for t, row in zip(times, values):
        assert cache.append(float(t), dict(zip(SIGS, row))) is True
    return cache


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 25, 31, 32, 33, 64, 100])
def test_downsampling_matches_an_independent_reference(n):
    times, values = random_rows(n, seed=n)
    cache = loaded_cache(times, values)
    for max_points in sorted({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, max(1, n - 1), n, n + 1}):
        df = cache.window(*EVERYTHING, max_points)
        t_ref, v_ref = bucket_reference(times, values, {2}, max_points)
        assert_typed(df)
        assert len(df) == len(t_ref) and len(df) <= max_points, (n, max_points)
        assert df.index.is_unique and df.index.is_monotonic_increasing
        np.testing.assert_allclose(epoch(df.index), t_ref, rtol=0, atol=TIME_ATOL_MEAN, err_msg=f"{n=} {max_points=}")
        np.testing.assert_allclose(df.to_numpy(), v_ref, rtol=1e-9, atol=1e-9, err_msg=f"{n=} {max_points=}")
        if n <= max_points:
            np.testing.assert_array_equal(df.to_numpy(), values)  # unchanged, not 'almost'
            np.testing.assert_allclose(epoch(df.index), times, rtol=0, atol=TIME_ATOL)


@pytest.mark.parametrize("flags", [(), ("flag",), ("b_analog", "flag"), ("flag", "b_analog"), SIGS, {"a_analog"}])
def test_downsampling_takes_the_maximum_for_exactly_the_flag_columns(flags):
    times, values = random_rows(37, seed=7)
    cache = loaded_cache(times, values, flags=flags)
    flag_columns = {i for i, name in enumerate(SIGS) if name in set(flags)}
    for max_points in (1, 5, 11, 36):
        df = cache.window(*EVERYTHING, max_points)
        _, v_ref = bucket_reference(times, values, flag_columns, max_points)
        np.testing.assert_allclose(df.to_numpy(), v_ref, rtol=1e-9, atol=1e-9, err_msg=f"{flags=} {max_points=}")


def test_downsampled_row_count_is_ceil_of_rows_over_bucket_size():
    for n in range(1, 41):
        cache = make(capacity=64)
        fill(cache, range(n))
        for max_points in sorted({1, 2, 3, 4, 5, 7, 8, 13, max(1, n - 1), n, n + 1}):
            df = cache.window(*EVERYTHING, max_points)
            k = -(-n // max_points) if n > max_points else 1
            assert len(df) == -(-n // k), (n, max_points)
            assert len(df) <= max_points, (n, max_points)


@pytest.mark.parametrize(
    "rows, max_points, expected_rows",
    [(10, 10, 10), (10, 9, 5), (10, 1, 1), (7, 6, 4), (10, 4, 4), (10, 3, 3), (9, 3, 3), (5000, 100, 100), (5001, 100, 99)],
)
def test_max_points_at_and_just_below_the_row_count(rows, max_points, expected_rows):
    cache = make(capacity=rows)
    fill(cache, range(rows))
    df = cache.window(*EVERYTHING, max_points)
    assert len(df) == expected_rows
    if rows <= max_points:
        assert ids(df) == list(range(rows))


def test_max_points_one_gives_a_single_row_with_the_global_mean_and_flag_maximum():
    times, values = random_rows(23, seed=3)
    df = loaded_cache(times, values).window(*EVERYTHING, 1)
    assert len(df) == 1
    np.testing.assert_allclose(epoch(df.index), [T0 + np.mean(times - T0)], rtol=0, atol=TIME_ATOL_MEAN)
    np.testing.assert_allclose(df.iloc[0].to_numpy(), [values[:, 0].mean(), values[:, 1].mean(), values[:, 2].max()], rtol=1e-9)


@pytest.mark.parametrize("at", [0, 457, 999])
@pytest.mark.parametrize("max_points", [1, 2, 7, 10, 33, 100, 999])
def test_a_single_sample_flag_pulse_survives_downsampling(at, max_points):
    n = 1000
    cache = make(capacity=n)
    for i in range(n):
        pulse = 1.0 if i == at else 0.0
        assert cache.append(stamp(i), {"b_analog": pulse, "a_analog": float(i), "flag": pulse}) is True
    df = cache.window(*EVERYTHING, max_points)
    k = -(-n // max_points)
    bucket = at // k
    size = min(k, n - bucket * k)
    flag = df["flag"].to_numpy()
    assert flag[bucket] == 1.0 and flag.sum() == 1.0  # the pulse stays at 1.0, not 1/k
    # the ordinary column with the same pulse is averaged
    assert df["b_analog"].to_numpy()[bucket] == pytest.approx(1.0 / size)


def test_flag_maximum_means_a_single_off_sample_is_not_preserved():
    cache = make(capacity=300)
    for i in range(300):
        assert cache.append(stamp(i), {"b_analog": 0.0, "a_analog": float(i), "flag": 0.0 if i == 150 else 1.0}) is True
    assert set(cache.window(*EVERYTHING, 10)["flag"]) == {1.0}


def test_flag_columns_take_the_maximum_even_for_negative_values():
    cache = make()
    for i, v in enumerate([-5.0, -3.0, -4.0, -9.0]):
        assert cache.append(stamp(i), {"b_analog": v, "a_analog": v, "flag": v}) is True
    df = cache.window(*EVERYTHING, 2)  # k = 2
    assert df["flag"].tolist() == [-3.0, -4.0]
    assert df["a_analog"].tolist() == [-4.0, -6.5]


def test_downsampling_buckets_start_at_the_first_qualifying_row():
    clock = FakeClock(stamp(20) + 20.0)
    cache = make(capacity=100, clock=clock, retention_s=20.0)  # retention lets rows through from 20 on
    fill(cache, range(50))
    # start lies before the retention bound: qualifying rows are 20..29 = 10 rows, k = 3 -> 3, 3, 3, 1
    assert cache.window(stamp(10), stamp(29), 4)["a_analog"].tolist() == [21.0, 24.0, 27.0, 29.0]
    # rows 20..49 = 30 rows -> k = 10
    assert cache.window(*EVERYTHING, 3)["a_analog"].tolist() == [24.5, 34.5, 44.5]
    # rows 25..34 = 10 rows -> k = 3 -> 3, 3, 3, 1; the buckets start at row 25, not at a multiple of k
    assert cache.window(stamp(25), stamp(34), 4)["a_analog"].tolist() == [26.0, 29.0, 32.0, 34.0]


def test_downsampled_frame_is_typed_like_any_other():
    times, values = random_rows(50, seed=5)
    df = loaded_cache(times, values).window(*EVERYTHING, 7)
    assert_typed(df)
    assert df.index.is_unique and df.index.is_monotonic_increasing
    assert len(df) <= 7


# ---- empty frames -----------------------------------------------------------------------------------


def test_an_empty_cache_gives_a_typed_empty_frame():
    cache = make()
    for max_points in (1, 10, UNLIMITED):
        assert_empty(cache.window(*EVERYTHING, max_points))
    assert_empty(cache.window(5.0, 1.0, 10))


def test_windows_without_rows_give_typed_empty_frames():
    clock = FakeClock(T0 + 100.0)
    cache = make(clock=clock, retention_s=1e6)
    fill(cache, range(10))
    assert_empty(cache.window(stamp(100), stamp(200), 10))  # after the data
    assert_empty(cache.window(stamp(-100), stamp(-50), 10))  # before the data
    assert_empty(cache.window(stamp(5) + 0.125, stamp(6) - 0.125, 10))  # between two rows
    assert_empty(cache.window(stamp(6), stamp(3), 10))  # start > end
    cache.clear()
    assert_empty(everything(cache))
    expired = make(clock=FakeClock(T0 + 10_000.0), retention_s=5.0)
    fill(expired, range(10))
    assert_empty(everything(expired))  # everything outside the retention


def test_empty_raw_and_downsampled_frames_all_have_the_same_dtypes():
    # Spec: 'so that code further down never sees an empty frame of a different type'; pandas 3 picks a different
    # resolution for an empty index than for a filled one, so the unit must be the same everywhere (which one
    # does not matter).
    cache = make()
    empty = everything(cache)
    fill(cache, range(10))
    others = [everything(cache), everything(cache, 3), cache.window(stamp(100), stamp(200), 5), cache.window(5.0, 1.0, 5)]
    for other in others:
        assert empty.index.dtype == other.index.dtype
        assert list(empty.dtypes) == list(other.dtypes)


@pytest.mark.parametrize("start, end", [(NAN, 1e18), (-1e18, NAN), (NAN, NAN)])
def test_window_with_a_nan_bound_is_empty_because_no_row_satisfies_the_comparison(start, end):
    cache = make()
    fill(cache, range(5))
    assert_empty(cache.window(start, end, 10))


def test_the_clock_is_called_without_holding_the_lock():
    # Spec: the lock is only held while data is being copied. A clock that itself queries the cache
    # (for example to keep up with the newest row) must therefore not make window() deadlock.
    holder = {}

    def clock() -> float:
        return holder["cache"].stats().newest + 1.0

    cache = SampleCache(SIGS, retention_s=60.0, max_samples=10, flags=FLAGS, clock=clock)
    holder["cache"] = cache
    fill(cache, range(5))
    outcome = {}

    def run():
        outcome["df"] = cache.window(*EVERYTHING, 100)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive(), "window() hangs: the clock was called while the lock was held"
    assert ids(outcome["df"]) == [0, 1, 2, 3, 4]


def test_empty_and_non_empty_frames_concatenate_without_changing_types():
    cache = make()
    empty = everything(cache)
    fill(cache, range(4))
    full = everything(cache)
    for combined in (pd.concat([empty, full]), pd.concat([full, empty])):
        assert_typed(combined)
        assert len(combined) == 4
        assert ids(combined) == [0, 1, 2, 3]


# ---- clear -----------------------------------------------------------------------------------------


def test_clear_removes_all_rows_and_keeps_the_capacity():
    cache = make(capacity=4)
    fill(cache, range(7))
    assert cache.clear() is None
    assert len(cache) == 0
    stats = cache.stats()
    assert (stats.rows, stats.capacity, stats.oldest, stats.newest) == (0, 4, None, None)
    assert_empty(everything(cache))


def test_clear_forgets_the_newest_timestamp():
    cache = make(step_back=5.0)
    fill(cache, range(10))
    cache.clear()
    before = cache.stats()
    assert cache.append(stamp(0), payload(0)) is True  # older than the former newest, but the cache is empty
    cache.clear()
    assert cache.append(stamp(-1000), payload(-1000)) is True  # also far back: there is nothing to jump back from
    after = cache.stats()
    assert after.dropped_out_of_order == before.dropped_out_of_order
    assert after.clock_resets == before.clock_resets
    assert_rows(everything(cache), [-1000])


def test_clear_resets_the_ring_position():
    cache = make(capacity=3)
    fill(cache, range(8))
    cache.clear()
    before = cache.stats()
    fill(cache, range(20, 24))  # 4 rows in capacity 3: wraps once
    assert_rows(everything(cache), [21, 22, 23])
    assert cache.stats().overwritten - before.overwritten == 1
    fill(cache, range(24, 30))
    assert_rows(everything(cache), [27, 28, 29])


def test_clear_on_an_empty_cache_is_harmless():
    cache = make(capacity=3)
    cache.clear()
    cache.clear()
    assert len(cache) == 0
    fill(cache, range(5))
    assert_rows(everything(cache), [2, 3, 4])


# ---- order: equal or older ---------------------------------------------------------------------


def test_equal_and_older_timestamps_are_dropped_as_out_of_order():
    cache = make(step_back=5.0)
    fill(cache, range(10))
    other = {"b_analog": -1.0, "a_analog": -2.0, "flag": 1.0}
    assert cache.append(stamp(9), other) is False  # equal counts too
    check_stats(cache, dropped_out_of_order=1)
    assert cache.append(stamp(8), other) is False
    assert cache.append(stamp(0), other) is False
    assert cache.append(stamp(9) - 5.0, other) is False  # exactly the threshold: still no reset
    check_stats(cache, dropped_out_of_order=4, dropped_invalid=0, clock_resets=0, rows=10, appended=10)
    assert_rows(everything(cache), range(10))  # the row with the equal time was not overwritten either
    assert cache.append(stamp(10), payload(10)) is True
    assert_rows(everything(cache), range(11))


def test_out_of_order_rows_never_disturb_a_full_ring():
    cache = make(capacity=3, step_back=5.0)
    fill(cache, range(6))
    before, overwritten = everything(cache), cache.stats().overwritten
    for i in (5, 4, 3, 0):
        assert cache.append(stamp(i), {"b_analog": 9.0, "a_analog": 9.0, "flag": 9.0}) is False
    pd.testing.assert_frame_equal(everything(cache), before)
    assert cache.stats().overwritten == overwritten
    assert ids(everything(cache)) == [3, 4, 5]


def test_a_one_ulp_step_forward_is_accepted_but_an_equal_timestamp_is_not():
    cache = make()
    t = stamp(0)
    assert cache.append(t, payload(0)) is True
    assert cache.append(t, payload(1)) is False
    assert cache.append(float(np.nextafter(t, np.inf)), payload(2)) is True
    assert cache.append(t, payload(3)) is False
    check_stats(cache, rows=2, appended=2, dropped_out_of_order=2, clock_resets=0)


# ---- clock jumping back --------------------------------------------------------------------------


def test_step_back_larger_than_the_threshold_resets_the_cache():
    cache = make(step_back=2.0)
    fill(cache, range(20))  # newest = stamp(19)
    new_ts = stamp(19) - 2.25  # 2.25 s back > 2.0
    assert cache.append(new_ts, payload(99)) is True
    assert len(cache) == 1
    df = everything(cache)
    assert ids(df) == [99]
    np.testing.assert_allclose(epoch(df.index), [new_ts], rtol=0, atol=TIME_ATOL)
    check_stats(
        cache, rows=1, oldest=new_ts, newest=new_ts, appended=21, clock_resets=1, dropped_out_of_order=0, overwritten=0
    )
    # after that everything simply continues, from the new timeline
    assert cache.append(new_ts + 0.25, payload(100)) is True
    assert cache.append(new_ts + 0.125, payload(101)) is False  # older than the new newest
    assert cache.append(stamp(30), payload(102)) is True
    assert ids(everything(cache)) == [99, 100, 102]
    check_stats(cache, clock_resets=1, dropped_out_of_order=1, appended=23, rows=3)


def test_step_back_exactly_at_the_threshold_is_not_a_reset():
    cache = make(step_back=2.0)
    fill(cache, range(20))
    assert cache.append(stamp(19) - 2.0, payload(99)) is False
    check_stats(cache, rows=20, clock_resets=0, dropped_out_of_order=1, appended=20)
    assert_rows(everything(cache), range(20))


def test_step_back_just_beyond_the_threshold_is_a_reset():
    cache = make(step_back=2.0)
    fill(cache, range(20))
    assert cache.append(stamp(19) - 2.0625, payload(99)) is True
    check_stats(cache, rows=1, clock_resets=1, dropped_out_of_order=0)
    assert ids(everything(cache)) == [99]


def test_step_back_threshold_is_configurable():
    cache = make(step_back=0.5)
    fill(cache, range(5))
    assert cache.append(stamp(4) - 0.5, payload(98)) is False
    assert cache.append(stamp(4) - 0.5625, payload(99)) is True
    check_stats(cache, clock_resets=1, rows=1)


def test_default_step_back_threshold_is_five_seconds():
    cache = SampleCache(SIGS, retention_s=1e6, max_samples=10, flags=FLAGS, clock=FakeClock())
    assert cache.append(T0 + 100.0, payload(0)) is True
    assert cache.append(T0 + 95.0, payload(1)) is False  # exactly 5.0 back
    check_stats(cache, clock_resets=0, dropped_out_of_order=1, rows=1)
    assert cache.append(T0 + 94.9375, payload(2)) is True  # 5.0625 back
    check_stats(cache, clock_resets=1, rows=1, oldest=T0 + 94.9375)


def test_step_back_is_measured_against_the_newest_stored_row_not_the_last_offered_one():
    cache = make(step_back=5.0)
    assert cache.append(T0 + 100.0, payload(0)) is True
    assert cache.append(T0 + 98.0, payload(1)) is False  # rejected: the newest stays T0 + 100
    assert cache.append(T0 + 94.5, payload(2)) is True  # 5.5 s below the newest: reset
    check_stats(cache, clock_resets=1, dropped_out_of_order=1, rows=1)
    assert ids(everything(cache)) == [2]


def test_reset_on_a_full_ring_clears_it_and_the_ring_keeps_working():
    cache = make(capacity=3, step_back=5.0)
    fill(cache, range(6))
    check_stats(cache, overwritten=3, rows=3)
    assert cache.append(stamp(-400), payload(-400)) is True
    check_stats(cache, rows=1, overwritten=3, clock_resets=1, appended=7)  # a reset is not an overwrite
    assert_rows(everything(cache), [-400])
    fill(cache, [-399, -398])
    check_stats(cache, rows=3, overwritten=3)
    assert_rows(everything(cache), [-400, -399, -398])
    fill(cache, [-397, -396])
    check_stats(cache, rows=3, overwritten=5, clock_resets=1)
    assert_rows(everything(cache), [-398, -397, -396])


def test_repeated_resets_are_counted_and_each_one_clears_the_cache():
    cache = make(capacity=10, step_back=5.0)
    fill(cache, range(1000, 1004))
    for resets, first in enumerate([500, 100, 0, -300], 1):  # each time more than 5 s below the newest row
        assert cache.append(stamp(first), payload(first)) is True
        check_stats(cache, clock_resets=resets, rows=1)
        assert ids(everything(cache)) == [first]
        fill(cache, [first + 1, first + 2])  # on the new timeline it simply continues
        assert ids(everything(cache)) == [first, first + 1, first + 2]
    check_stats(cache, clock_resets=4, dropped_out_of_order=0, dropped_invalid=0)


def test_reset_on_a_cache_with_one_row():
    cache = make(step_back=5.0)
    fill(cache, [100])
    assert cache.append(stamp(100), payload(1)) is False  # equal: no reset
    assert cache.append(stamp(100) - 6.0, payload(2)) is True
    check_stats(cache, clock_resets=1, dropped_out_of_order=1, rows=1, appended=2)
    assert ids(everything(cache)) == [2]


def test_an_invalid_row_never_triggers_a_reset_or_an_out_of_order_count():
    cache = make(step_back=5.0)
    fill(cache, range(10))
    for bad in (payload(1) | {"flag": NAN}, {"b_analog": 1.0, "a_analog": 1.0}):
        assert cache.append(stamp(9) - 100.0, bad) is False  # far back, but invalid
        assert cache.append(stamp(9) - 1.0, bad) is False  # within the threshold, but invalid
    check_stats(cache, dropped_invalid=4, dropped_out_of_order=0, clock_resets=0, rows=10, appended=10)
    assert_rows(everything(cache), range(10))


# ---- copy semantics -------------------------------------------------------------------------------


def write_through(array: np.ndarray) -> None:
    array.setflags(write=True)  # pandas returns read-only views; forcing writes exposes a shared buffer
    array[...] = 0


def mutate_every_way(df: pd.DataFrame) -> None:
    """Try to modify the frame in every way. Any attempt may fail; nothing may touch the cache."""
    attempts = [
        lambda: df.iloc.__setitem__((slice(None), slice(None)), -1.0),
        lambda: df.__setitem__("a_analog", 12345.0),
        lambda: df.loc.__setitem__((df.index[0], "flag"), 777.0),
        lambda: df.sort_index(ascending=False, inplace=True),
        lambda: df.reset_index(drop=True, inplace=True),
        lambda: df.drop(columns=["flag"], inplace=True),
        lambda: write_through(df.to_numpy()),
        lambda: write_through(df["a_analog"].to_numpy()),
        lambda: write_through(df.index.asi8),
        lambda: write_through(np.asarray(df)),
    ]
    for attempt in attempts:
        try:
            attempt()
        except Exception:  # noqa: BLE001 - read-only is fine
            pass


@pytest.mark.parametrize("max_points", [UNLIMITED, 3])
def test_modifying_a_returned_frame_never_affects_the_cache(max_points):
    cache = make(capacity=10)
    fill(cache, range(8))
    pristine = everything(cache, max_points)
    victim = everything(cache, max_points)
    mutate_every_way(victim)
    pd.testing.assert_frame_equal(everything(cache, max_points), pristine)
    assert len(cache) == 8


def test_two_windows_are_independent_frames():
    cache = make(capacity=10)
    fill(cache, range(6))
    first, second = everything(cache), everything(cache)
    pristine = second.copy(deep=True)
    mutate_every_way(first)
    pd.testing.assert_frame_equal(second, pristine)


@pytest.mark.parametrize("max_points", [UNLIMITED, 2])
def test_a_returned_frame_is_not_changed_by_later_appends(max_points):
    cache = make(capacity=3)
    fill(cache, range(3))
    frame = everything(cache, max_points)
    pristine = frame.copy(deep=True)
    fill(cache, range(3, 25))  # the ring goes all the way round and overwrites every position
    cache.clear()
    fill(cache, range(100, 110))
    pd.testing.assert_frame_equal(frame, pristine)


# ---- stats ------------------------------------------------------------------------------------------


def test_stats_report_the_oldest_and_newest_stored_row_not_those_of_a_window():
    cache = make(capacity=5)
    fill(cache, range(12))
    assert ids(cache.window(stamp(9), stamp(10), 10)) == [9, 10]
    check_stats(cache, oldest=stamp(7), newest=stamp(11), rows=5, appended=12, overwritten=7)


def test_stats_object_is_a_snapshot():
    cache = make()
    fill(cache, range(3))
    first = cache.stats()
    fill(cache, range(3, 6))
    second = cache.stats()
    assert (first.rows, first.appended, first.newest) == (3, 3, pytest.approx(stamp(2)))
    assert (second.rows, second.appended, second.newest) == (6, 6, pytest.approx(stamp(5)))


# ---- random sequences against a naive reference model -----------------------------------------


class ReferenceModel:
    """The spec executed literally on a plain list: no ring buffer, no numpy magic."""

    def __init__(self, signals, retention_s, max_samples, flags, step_back):
        self.signals = list(signals)
        self.retention_s, self.capacity, self.step_back = retention_s, max_samples, step_back
        self.flag_columns = {i for i, s in enumerate(self.signals) if s in set(flags)}
        self.t, self.v = [], []
        self.appended = self.out_of_order = self.invalid = self.overwritten = self.resets = 0

    def append(self, timestamp, values):
        try:
            row = [values[s] for s in self.signals]
            ok = math.isfinite(timestamp) and all(math.isfinite(x) for x in row)
        except KeyError:
            ok = False
        if not ok:
            self.invalid += 1
            return False
        if self.t and timestamp <= self.t[-1]:
            if self.t[-1] - timestamp > self.step_back:
                self.t.clear()
                self.v.clear()
                self.resets += 1
            else:
                self.out_of_order += 1
                return False
        if len(self.t) == self.capacity:
            del self.t[0], self.v[0]
            self.overwritten += 1
        self.t.append(timestamp)
        self.v.append(row)
        self.appended += 1
        return True

    def expected_stats(self):
        return {
            "rows": len(self.t),
            "capacity": self.capacity,
            "oldest": self.t[0] if self.t else None,
            "newest": self.t[-1] if self.t else None,
            "appended": self.appended,
            "dropped_out_of_order": self.out_of_order,
            "dropped_invalid": self.invalid,
            "overwritten": self.overwritten,
            "clock_resets": self.resets,
        }

    def window(self, start, end, max_points, now):
        cutoff = now - self.retention_s
        keep = [i for i, t in enumerate(self.t) if start <= t <= end and t >= cutoff]
        times = np.array([self.t[i] for i in keep], dtype=float)
        values = np.array([self.v[i] for i in keep], dtype=float).reshape(len(keep), len(self.signals))
        return bucket_reference(times, values, self.flag_columns, max_points)


MODEL_CONFIGS = [
    # seed, capacity, retention, step threshold, flags
    (1, 2, 1e6, 2.0, ("flag",)),
    (2, 3, 8.0, 0.5, ("flag",)),
    (3, 5, 30.0, 5.0, ("flag", "b_analog")),
    (4, 17, 30.0, 1.0, ()),
    (5, 64, 1e6, 5.0, ("a_analog",)),
    (6, 7, 4.0, 0.25, SIGS),
]
MODEL_STEPS = 600


@pytest.mark.parametrize("seed, capacity, retention_s, step_back, flags", MODEL_CONFIGS)
def test_random_operation_sequences_match_a_reference_model(seed, capacity, retention_s, step_back, flags):
    rng = random.Random(seed)
    clock = FakeClock(T0)
    cache = SampleCache(
        SIGS, retention_s=retention_s, max_samples=capacity, flags=flags, clock=clock, clock_step_back_s=step_back
    )
    model = ReferenceModel(SIGS, retention_s, capacity, flags, step_back)
    back_steps = int(step_back / 0.125)

    for step in range(MODEL_STEPS):
        newest = model.t[-1] if model.t else None
        base = newest if newest is not None else T0 + 50.0
        choice = rng.random()
        if choice < 0.62 or newest is None:
            ts = base + rng.choice([1, 2, 3, 5]) * 0.125
        elif choice < 0.70:
            ts = newest  # equal
        elif choice < 0.80:
            ts = newest - rng.randint(1, back_steps) * 0.125  # back, up to and including exactly the threshold
        elif choice < 0.86:
            ts = newest - step_back - rng.randint(1, 40) * 0.125  # beyond the threshold: reset
        else:
            ts = base + rng.choice([1, 2]) * 0.125
        values = {
            name: rng.choice([0.0, 1.0, 1.0, 2.5, -1.0]) if name in flags else round(rng.uniform(-500.0, 500.0), 3)
            for name in rng.sample(SIGS, k=len(SIGS))  # random key order
        }
        spoil = rng.random()
        if spoil < 0.04:
            ts = rng.choice([NAN, INF, -INF])
        elif spoil < 0.08:
            values[rng.choice(SIGS)] = rng.choice([NAN, INF, -INF])
        elif spoil < 0.11:
            del values[rng.choice(SIGS)]
        if rng.random() < 0.15:
            values["junk"] = rng.choice([NAN, INF, 1.0])

        assert cache.append(ts, values) is model.append(ts, values), (step, ts)
        assert len(cache) == len(model.t)
        stats = dataclasses.asdict(cache.stats())
        expected = model.expected_stats()
        for name in ("oldest", "newest"):
            assert stats.pop(name) == pytest.approx(expected.pop(name), abs=TIME_ATOL), (step, name)
        assert stats == expected, step

        if step % 2 == 0:
            anchor = model.t[-1] if model.t else T0
            clock.now = anchor + rng.choice([0.0, 0.125, 1.5, 7.25, 20.0, 100.0])
            start = rng.choice([-1e18, anchor - rng.uniform(0, 60), anchor - rng.randint(0, 100) * 0.125])
            end = rng.choice([1e18, anchor + 1.0, anchor - rng.randint(0, 100) * 0.125])
            max_points = rng.choice([1, 2, 3, 4, 5, 7, 10, 50, 1000])
            df = cache.window(start, end, max_points)
            t_ref, v_ref = model.window(start, end, max_points, clock.now)
            where = (seed, step, start, end, max_points)
            assert_typed(df)
            assert len(df) == len(t_ref), where
            np.testing.assert_allclose(epoch(df.index), t_ref, rtol=0, atol=TIME_ATOL_MEAN, err_msg=str(where))
            np.testing.assert_allclose(df.to_numpy(), v_ref, rtol=1e-9, atol=1e-9, err_msg=str(where))


# ---- fixed memory and O(1) -------------------------------------------------------------------------


def traced_bytes() -> int:
    gc.collect()
    return tracemalloc.get_traced_memory()[0]


def test_constructor_allocates_the_float64_arrays_up_front():
    capacity, signals = 100_000, ["a", "b", "c"]
    tracemalloc.start()
    try:
        before = traced_bytes()
        cache = SampleCache(signals, retention_s=60.0, max_samples=capacity, clock=FakeClock())
        allocated = traced_bytes() - before
    finally:
        tracemalloc.stop()
    assert allocated >= 0.95 * 8 * capacity * (1 + len(signals))  # times[max_samples] + values[max_samples, n]
    assert len(cache) == 0


def test_a_flood_of_appends_never_grows_memory():
    cache = make(capacity=1_000)
    fill(cache, range(1_500))  # ring full and already wrapped
    last = 21_499
    tracemalloc.start()
    try:
        before = traced_bytes()
        for i in range(1_500, last + 1):
            cache.append(stamp(i), payload(i))  # valid: 20 times around the ring
        for _ in range(3_000):
            cache.append(stamp(last), payload(0))  # equal time: ordering error
        for _ in range(3_000):
            cache.append(NAN, payload(0))  # invalid
        grown = traced_bytes() - before
    finally:
        tracemalloc.stop()
    assert grown < 32 * 1024  # a retained row or list would be hundreds of kB here
    check_stats(cache, rows=1_000, capacity=1_000, appended=last + 1, dropped_out_of_order=3_000, dropped_invalid=3_000)


def test_append_allocates_nothing_proportional_to_the_capacity():
    capacity = 20_000
    cache = make(capacity=capacity)
    fill(cache, range(capacity + 100))  # full and wrapped
    rows = [(stamp(i), payload(i)) for i in range(capacity + 100, capacity + 400)]
    gc.collect()
    tracemalloc.start()
    try:
        baseline = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        for t, row in rows:
            assert cache.append(t, row) is True
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak - baseline < 16 * 1024  # a copy of the time or value series is 160 kB or more
    assert len(cache) == capacity


# ---- threads ----------------------------------------------------------------------------------------

GRID = 64.0  # grid points per second: row number and time can thus be converted into each other exactly
STRESS_SIGNALS = ("b", "flag", "a")
STRESS_CAPACITY = 200
STRESS_APPENDS = 3_000
STRESS_PERIOD = 600  # every so many appends the writer jumps back (reset) or clears (clear)
MIN_READS = 15


def grid_time(i: int) -> float:
    return T0 + i / GRID


def grid_values(i: int) -> dict[str, float]:
    return {"a": float(i), "b": 3.0 * i + 1.0, "flag": 1.0 if i % 10 == 0 else 0.0}


def check_stress_frame(df: pd.DataFrame, max_points: int) -> None:
    """A frame is a consistent snapshot: time strictly increasing, the columns of a row belong together."""
    assert len(df) <= max_points
    assert list(df.columns) == list(STRESS_SIGNALS)
    assert_utc_index(df.index)
    if df.empty:
        return
    assert df.index.is_unique and df.index.is_monotonic_increasing, "time not strictly increasing"
    t, a, b, flag = epoch(df.index), df["a"].to_numpy(), df["b"].to_numpy(), df["flag"].to_numpy()
    np.testing.assert_allclose(
        b, 3.0 * a + 1.0, rtol=1e-9, atol=1e-6, err_msg="columns of a row do not belong together"
    )
    np.testing.assert_allclose(t, T0 + a / GRID, rtol=0, atol=TIME_ATOL_MEAN, err_msg="time does not match the values")
    assert set(np.unique(flag)) <= {0.0, 1.0}
    if len(df) < 2:
        return
    steps = np.diff(a)
    k = steps[0]
    # The snapshot is contiguous: all full buckets are equally far apart, the last (shorter) one at most.
    assert k >= 1 and np.allclose(steps[:-1], k) and 0 < steps[-1] <= k + 1e-9, steps.tolist()
    if len(df) < 3:
        return
    size = int(round(k))
    first = a[:-1] - (size - 1) / 2.0  # first row of every full bucket
    assert np.allclose(first, np.rint(first))
    first = np.rint(first).astype(np.int64)
    has_multiple_of_10 = ((first + size - 1) // 10) > ((first - 1) // 10)  # the flag is 1 for row numbers 0, 10, 20...
    assert np.array_equal(flag[:-1] > 0.5, has_multiple_of_10), "flag does not belong with the rows of the bucket"
    if size == 1:  # raw rows: the last row is then a 'full bucket' too
        assert (flag[-1] > 0.5) == (a[-1] % 10 == 0)


def check_stress_stats(stats: CacheStats, scenario: str) -> None:
    assert 0 <= stats.rows <= STRESS_CAPACITY
    assert (stats.rows == 0) == (stats.oldest is None) == (stats.newest is None)
    if stats.rows:
        assert stats.oldest <= stats.newest
    if scenario == "plain":  # without reset or clear, rows is exactly stored minus overwritten
        assert stats.rows == stats.appended - stats.overwritten


def reader_loop(cache, scenario, stop, failures, counts, slot, deadline):
    choices = [UNLIMITED, 50, 7, 1, UNLIMITED, 120]
    reads = 0
    try:
        while (not stop.is_set() or reads < MIN_READS) and time.monotonic() < deadline:
            max_points = choices[reads % len(choices)]
            check_stress_frame(cache.window(*EVERYTHING, max_points), max_points)
            check_stress_stats(cache.stats(), scenario)
            assert 0 <= len(cache) <= STRESS_CAPACITY
            reads += 1
    except BaseException as exc:  # noqa: BLE001 - any error in the thread must make the test fail
        failures.append(f"reader {slot}: {type(exc).__name__}: {exc}")
        stop.set()
    counts[slot] = reads


def writer_loop(cache, scenario, stop, failures, latest, last_row):
    i = 5_000
    try:
        for step in range(STRESS_APPENDS):
            if step and step % STRESS_PERIOD == 0:
                if scenario == "resets":
                    i -= 800  # 12.5 s back: beyond the 5 s threshold, so a reset
                elif scenario == "clears":
                    cache.clear()
                    i += 500  # gap in the series: a mix of rows from before and after the clear then stands out
            if cache.append(grid_time(i), grid_values(i)) is not True:
                raise AssertionError(f"row {i} not stored")
            latest[0] = grid_time(i)
            last_row[0] = i
            i += 1
    except BaseException as exc:  # noqa: BLE001
        failures.append(f"writer: {type(exc).__name__}: {exc}")
    finally:
        stop.set()


@pytest.mark.parametrize("scenario", ["plain", "resets", "clears"])
def test_readers_see_consistent_frames_while_the_writer_appends(scenario):
    latest, last_row = [grid_time(5_000)], [None]
    if scenario == "plain":
        clock, retention_s = (lambda: latest[0]), 3.0  # clock keeps up with the writer; the retention cuts into the ring
    else:
        clock, retention_s = FakeClock(T0), 1e12
    cache = SampleCache(
        STRESS_SIGNALS,
        retention_s=retention_s,
        max_samples=STRESS_CAPACITY,
        flags=("flag",),
        clock=clock,
        clock_step_back_s=5.0,
    )
    stop, failures, counts = threading.Event(), [], [0] * 4
    deadline = time.monotonic() + 40
    readers = [
        threading.Thread(target=reader_loop, args=(cache, scenario, stop, failures, counts, n, deadline), daemon=True)
        for n in range(4)
    ]
    writer = threading.Thread(target=writer_loop, args=(cache, scenario, stop, failures, latest, last_row), daemon=True)
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)  # switch threads much more often: that makes missing lock usage visible
    try:
        for thread in readers:
            thread.start()
        writer.start()
        writer.join(timeout=30)
        for thread in readers:
            thread.join(timeout=15)
    finally:
        sys.setswitchinterval(interval)
        stop.set()
    assert not writer.is_alive(), "writer hangs: deadlock?"
    assert not any(thread.is_alive() for thread in readers), "a reader hangs: deadlock?"
    assert not failures, failures[:3]
    assert min(counts) >= MIN_READS, counts

    check_stress_frame(everything(cache), UNLIMITED)
    stats = cache.stats()
    assert stats.rows == STRESS_CAPACITY
    assert stats.newest == pytest.approx(grid_time(last_row[0]), abs=TIME_ATOL)
    if scenario == "plain":
        assert (stats.appended, stats.overwritten) == (STRESS_APPENDS, STRESS_APPENDS - STRESS_CAPACITY)
        assert (stats.dropped_out_of_order, stats.dropped_invalid, stats.clock_resets) == (0, 0, 0)
    elif scenario == "resets":
        assert stats.appended == STRESS_APPENDS and stats.clock_resets == (STRESS_APPENDS - 1) // STRESS_PERIOD


PANDAS_DIR = os.path.dirname(os.path.abspath(pd.__file__)) + os.sep
NUMPY_DIR = os.path.dirname(os.path.abspath(np.__file__)) + os.sep
# Computation that does not count as 'only copying data': the usual numpy reductions for a mean or maximum
NUMPY_REDUCTIONS = frozenset({"mean", "_mean", "sum", "_sum", "amax", "_amax", "amin", "_amin", "average", "median", "std", "var"})


def is_probe_point(frame, event, arg) -> str | None:
    """Name of the pandas or numpy computation that starts here, or None."""
    if event == "call":
        filename = frame.f_code.co_filename
        if filename.startswith(PANDAS_DIR):
            return frame.f_code.co_name
        if filename.startswith(NUMPY_DIR) and frame.f_code.co_name in NUMPY_REDUCTIONS:
            return frame.f_code.co_name
    elif event == "c_call" and getattr(arg, "__name__", "") in ("reduceat", "reduce"):
        if isinstance(getattr(arg, "__self__", None), np.ufunc):  # np.add.reduceat(...), np.maximum.reduce(...)
            return arg.__name__
    return None


def probe_lock_during_processing(window_call, append_call, timeout_s=2.0, max_handshakes=6_000):
    """Make a reader stop in the middle of pandas or numpy computation (window_call) and see whether a writer can
    still append.

    A profile hook in the reader waits at the start of every such piece of work until a second thread has run
    append_call. If the cache holds its lock at that moment, the writer does not get through and the wait times out.
    """
    go, done = threading.Event(), threading.Event()
    state = {"calls": 0, "blocked": None, "stop": False, "quit": False, "errors": []}

    def worker():
        while True:
            go.wait()
            go.clear()
            if state["quit"]:
                return
            try:
                append_call()
            except BaseException as exc:  # noqa: BLE001
                state["errors"].append(repr(exc))
            finally:
                done.set()

    def hook(frame, event, arg):
        if state["stop"]:
            return
        name = is_probe_point(frame, event, arg)
        if name is None:
            return
        state["calls"] += 1
        if state["calls"] > max_handshakes:
            state["stop"] = True
            return
        done.clear()
        go.set()
        if not done.wait(timeout_s):
            state["blocked"] = name
            state["stop"] = True

    def reader():
        sys.setprofile(hook)
        try:
            window_call()
        except BaseException as exc:  # noqa: BLE001
            state["errors"].append(repr(exc))
        finally:
            sys.setprofile(None)

    helper, reader_thread = threading.Thread(target=worker, daemon=True), threading.Thread(target=reader, daemon=True)
    helper.start()
    reader_thread.start()
    reader_thread.join(timeout=30)
    done.wait(timeout=5)  # a writer that was waiting for the lock may finish now
    state["quit"] = True
    go.set()
    helper.join(timeout=5)
    state["hung"] = reader_thread.is_alive() or helper.is_alive()
    return state


@pytest.mark.parametrize("max_points", [UNLIMITED, 20])
def test_the_lock_is_held_only_while_data_is_copied_not_while_the_frame_is_built_or_downsampled(max_points):
    cache = make(capacity=200)
    fill(cache, range(120))
    counter = itertools.count(1_000)

    def append_call():
        i = next(counter)
        assert cache.append(stamp(i), payload(i)) is True

    state = probe_lock_during_processing(lambda: cache.window(*EVERYTHING, max_points), append_call)
    assert not state["hung"], "deadlock between reader and writer"
    assert state["errors"] == []
    assert state["calls"] > 0, "the probe saw no pandas or numpy computation: the test measures nothing"
    assert state["blocked"] is None, f"append kept hanging while window() was in {state['blocked']}"


class LockHoldingCache:
    """Deliberately wrong: computation under the lock. Proves that the probe really finds such a mistake."""

    def __init__(self, work):
        self.lock = threading.Lock()
        self.work = work

    def append(self):
        with self.lock:
            return True

    def window(self):
        with self.lock:
            return self.work()


@pytest.mark.parametrize(
    "work",
    [
        lambda: pd.DataFrame({"x": [1.0, 2.0]}, index=pd.to_datetime([1, 2], unit="s", utc=True)),
        lambda: np.add.reduceat(np.arange(8.0), np.array([0, 4])),
        lambda: np.arange(8.0).mean(),
    ],
    ids=["pandas", "numpy-reduceat", "numpy-mean"],
)
def test_the_probe_does_detect_processing_under_the_lock(work):
    bad = LockHoldingCache(work)
    state = probe_lock_during_processing(bad.window, bad.append, timeout_s=0.1)  # a really blocked append never finishes earlier
    assert state["calls"] > 0
    assert state["blocked"] is not None
    assert not state["hung"]
