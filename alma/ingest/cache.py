"""In-memory cache holding the last few minutes of live samples.

The UDP receiver writes (one thread), the dashboard reads (several threads). The lock is only held while data is
being copied; all pandas work happens outside it, so a slow reader never holds up the receiver (with UDP, a telegram
that is left waiting is simply lost).
"""

import math
import operator
import threading
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

# datetime64[ns] reaches up to ~2262. A timestamp beyond that cannot become a DataFrame index and would make
# every window() fail for as long as that row sits in the ring, so we reject it on append.
_TS_LIMIT_S = 9_223_372_036.0
_NS_PER_S = 1_000_000_000


@dataclass(frozen=True)
class CacheStats:
    rows: int
    capacity: int
    oldest: float | None  # epoch seconds of the oldest and the newest stored row; None if the cache is empty
    newest: float | None
    appended: int  # rows stored since start; clear() and a clock reset do not reset the counters
    dropped_out_of_order: int
    dropped_invalid: int
    overwritten: int  # rows that left the full ring because a newer one arrived
    clock_resets: int


def _finite(x) -> float | None:
    """x as a finite float, or None for anything that is not (including None and text): append() must never raise."""
    if isinstance(x, (str, bytes, bytearray)):
        return None  # float("1.5") would succeed, but a decoder that delivers text has a bug
    try:
        number = float(x)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _as_float(x) -> float:
    """float(x), or NaN if that is not possible; the caller rejects NaN anyway."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


class SampleCache:
    """Ring buffer of timestamped rows (one value per signal), strictly increasing in time.

    Memory is fixed at construction: a stream that arrives faster or for longer than expected
    overwrites the oldest rows and lets nothing grow.

    One writer and several readers at the same time is safe. Every window that is read out comes from a single
    moment of the ring, so the values in a row always belong to the time of that row.
    """

    def __init__(
        self,
        signals: Sequence[str],
        *,
        retention_s: float,
        max_samples: int,
        flags: Collection[str] = (),
        clock: Callable[[], float] = time.time,
        clock_step_back_s: float = 5.0,
    ) -> None:
        if isinstance(signals, (str, bytes)):
            raise ValueError("signals must be a list of names, not a single string")
        names = tuple(signals)
        if not names:
            raise ValueError("signals must not be empty")
        if not all(isinstance(name, str) for name in names):
            raise ValueError("signals must only contain string names")
        doubles = sorted({name for name in names if names.count(name) > 1})
        if doubles:
            raise ValueError(f"signals must be unique, duplicates: {', '.join(doubles)}")

        retention = _as_float(retention_s)
        if not (math.isfinite(retention) and retention > 0):
            raise ValueError(f"retention_s must be a finite number greater than 0, got {retention_s!r}")
        try:
            capacity = operator.index(max_samples)
        except TypeError:
            raise ValueError(f"max_samples must be an integer, got {max_samples!r}") from None
        if capacity < 2:
            raise ValueError(f"max_samples must be at least 2, got {capacity}")
        step_back = _as_float(clock_step_back_s)
        if not step_back > 0:  # 'not' also catches NaN: a threshold that never becomes true silences the clock reset
            raise ValueError(f"clock_step_back_s must be greater than 0, got {clock_step_back_s!r}")
        if isinstance(flags, (str, bytes)):
            raise ValueError("flags must be a collection of names, not a single string")
        unknown = sorted(str(flag) for flag in set(flags) - set(names))
        if unknown:
            raise ValueError(f"flags contains names that are not in signals: {', '.join(unknown)}")
        if not callable(clock):
            raise TypeError("clock must be callable and return epoch seconds")

        self._signals = names
        self._columns = list(names)
        self._width = len(names)
        self._capacity = capacity
        self._retention_s = retention
        self._step_back_s = step_back
        self._clock = clock
        flagged = set(flags)
        self._flag_cols = np.array([i for i, name in enumerate(names) if name in flagged], dtype=np.intp)

        # Allocated once; nothing grows from here on, no matter how much arrives.
        self._times = np.zeros(capacity)
        self._values = np.zeros((capacity, self._width))

        self._lock = threading.Lock()
        self._head = 0  # next write position; the oldest row sits at (head - count) % capacity
        self._count = 0
        self._newest: float | None = None
        self._appended = self._dropped_out_of_order = self._dropped_invalid = 0
        self._overwritten = self._clock_resets = 0

    def append(self, timestamp: float, values: Mapping[str, float]) -> bool:
        """Store a row. True if it was stored, False if it was discarded; never raises on bad data."""
        ts = _finite(timestamp)
        row = self._row(values) if ts is not None and -_TS_LIMIT_S <= ts <= _TS_LIMIT_S else None
        if row is None:
            with self._lock:
                self._dropped_invalid += 1
            return False

        with self._lock:
            newest = self._newest
            if newest is not None and ts <= newest:
                if newest - ts <= self._step_back_s:
                    self._dropped_out_of_order += 1
                    return False
                # System clock set back (NTP). Without clearing, the cache would reject every telegram until the
                # clock reaches the old level again, and that can take hours.
                self._reset_locked()
                self._clock_resets += 1
            self._store_locked(ts, row)
        return True

    def window(self, start: float, end: float, max_points: int) -> pd.DataFrame:
        """Rows with start <= t <= end that fall within the retention, increasing in time, as a copy.

        If there are more than max_points, consecutive buckets of k = ceil(rows / max_points)
        rows are merged: mean time, mean value, and the maximum for the flag columns
        (so that a short ON pulse does not disappear).
        """
        try:
            limit = operator.index(max_points)
        except TypeError:
            raise ValueError(f"max_points must be an integer, got {max_points!r}") from None
        if limit < 1:
            raise ValueError(f"max_points must be at least 1, got {limit}")
        start, end = float(start), float(end)
        if not start <= end:  # also if either of them is NaN
            return self._empty()

        # The clock is read outside the lock: a clock that itself queries the cache must not deadlock.
        bound = self._clock() - self._retention_s
        low = start if start >= bound else bound  # with a NaN clock this becomes NaN and the window stays empty
        with self._lock:
            first, last = self._locate(low, end)
            rows = self._copy_rows(first, last) if last > first else None
        if rows is None:
            return self._empty()

        times, values = rows
        if len(times) > limit:
            times, values = self._downsample(times, values, limit)
        return self._frame(times, values)

    def clear(self) -> None:
        """Discard all rows. The counters stay: they cover the lifetime of the cache."""
        with self._lock:
            self._reset_locked()

    def stats(self) -> CacheStats:
        """Counters and extreme times of this single moment; all read under one lock, so they agree with each other."""
        with self._lock:
            count = self._count
            oldest = float(self._times[(self._head - count) % self._capacity]) if count else None
            return CacheStats(
                rows=count,
                capacity=self._capacity,
                oldest=oldest,
                newest=self._newest,
                appended=self._appended,
                dropped_out_of_order=self._dropped_out_of_order,
                dropped_invalid=self._dropped_invalid,
                overwritten=self._overwritten,
                clock_resets=self._clock_resets,
            )

    def __len__(self) -> int:
        """Number of stored rows, regardless of retention (which only applies when reading)."""
        with self._lock:
            return self._count

    # ---- storing --------------------------------------------------------------------------------------

    def _row(self, values: Mapping[str, float]) -> np.ndarray | None:
        """The values in column order, or None if one is missing or not finite. Outside the lock,
        and fully checked before anything is stored: a half-valid row never gets in."""
        row = []
        try:
            for name in self._signals:
                value = _finite(values[name])
                if value is None:
                    return None
                row.append(value)
        except (KeyError, TypeError, IndexError):  # signal is missing, or values is not even a mapping
            return None
        return np.array(row)

    def _store_locked(self, ts: float, row: np.ndarray) -> None:
        slot = self._head
        self._times[slot] = ts
        self._values[slot] = row
        self._head = (slot + 1) % self._capacity
        if self._count < self._capacity:
            self._count += 1
        else:
            self._overwritten += 1
        self._newest = ts
        self._appended += 1

    def _reset_locked(self) -> None:
        self._head = 0
        self._count = 0
        self._newest = None

    # ---- reading -------------------------------------------------------------------------------------

    def _locate(self, low: float, high: float) -> tuple[int, int]:
        """Logical positions [first, last) (0 = oldest row) of the rows with low <= t <= high.

        The ring is at most two contiguous pieces that are increasing together, so two binary
        searches per bound suffice and nothing needs to be copied from the ring to search it.
        """
        count = self._count
        if not count:
            return 0, 0
        oldest = (self._head - count) % self._capacity
        head_len = min(count, self._capacity - oldest)
        head, tail = self._times[oldest : oldest + head_len], self._times[: count - head_len]

        def position(x: float, side: str) -> int:
            k = int(head.searchsorted(x, side=side))
            if k == head_len and len(tail):  # x lies beyond the first piece, so keep searching in the second
                k += int(tail.searchsorted(x, side=side))
            return k

        return position(low, "left"), position(high, "right")

    def _copy_rows(self, first: int, last: int) -> tuple[np.ndarray, np.ndarray]:
        """Copy of the logical rows [first, last); only this part happens under the lock."""
        begin = (self._head - self._count + first) % self._capacity
        stop = begin + (last - first)
        if stop <= self._capacity:
            return self._times[begin:stop].copy(), self._values[begin:stop].copy()
        wrapped = stop - self._capacity
        return (
            np.concatenate((self._times[begin:], self._times[:wrapped])),
            np.concatenate((self._values[begin:], self._values[:wrapped])),
        )

    def _downsample(self, times: np.ndarray, values: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
        rows = len(times)
        size = -(-rows // limit)  # ceiling without float rounding
        starts = np.arange(0, rows, size)
        counts = np.diff(np.append(starts, rows))  # the last bucket may be shorter

        firsts = times[starts]
        # Average times relative to the start of their bucket: a sum of epoch seconds (~1.7e9) would lose digits.
        mean_times = firsts + np.add.reduceat(times - np.repeat(firsts, counts), starts) / counts
        out = np.add.reduceat(values, starts, axis=0) / counts[:, None]
        if len(self._flag_cols):
            out[:, self._flag_cols] = np.maximum.reduceat(values[:, self._flag_cols], starts, axis=0)
        return mean_times, out

    def _empty(self) -> pd.DataFrame:
        return self._frame(np.empty(0), np.empty((0, self._width)))

    def _frame(self, times: np.ndarray, values: np.ndarray) -> pd.DataFrame:
        # Build the index ourselves with a fixed unit: pandas 3 picks a different unit for an empty index
        # than for a filled one, and then an empty window would have a different type than a filled one.
        seconds = np.floor(times)
        nanos = seconds.astype(np.int64) * _NS_PER_S + np.rint((times - seconds) * _NS_PER_S).astype(np.int64)
        index = pd.DatetimeIndex(nanos.astype("datetime64[ns]")).tz_localize("UTC")
        # copy=False: times and values are already our own copy, another copy by pandas would be a waste.
        return pd.DataFrame(values, index=index, columns=self._columns, copy=False)
