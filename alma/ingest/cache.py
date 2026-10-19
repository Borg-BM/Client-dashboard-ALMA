"""Geheugencache met de laatste minuten live-samples.

De UDP-ontvanger schrijft (een thread), het dashboard leest (meerdere threads). Het slot wordt alleen
vastgehouden terwijl er data wordt overgenomen; al het pandas-werk gebeurt erbuiten, zodat een trage
lezer de ontvanger nooit ophoudt (een telegram dat blijft liggen is bij UDP gewoon verloren).
"""

import math
import operator
import threading
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

# datetime64[ns] reikt tot ~2262. Een tijdstempel daarbuiten kan geen DataFrame-index worden en zou
# elke window() laten falen zolang die rij in de ring staat; dus weigeren we hem bij het opslaan.
_TS_LIMIT_S = 9_223_372_036.0
_NS_PER_S = 1_000_000_000


@dataclass(frozen=True)
class CacheStats:
    rows: int
    capacity: int
    oldest: float | None  # epoch-seconden van de oudste en de nieuwste opgeslagen rij; None als de cache leeg is
    newest: float | None
    appended: int  # opgeslagen rijen sinds de start; clear() en een klokherstel zetten de tellers niet terug
    dropped_out_of_order: int
    dropped_invalid: int
    overwritten: int  # rijen die uit de volle ring verdwenen omdat er een nieuwere kwam
    clock_resets: int


def _finite(x) -> float | None:
    """x als eindige float, of None bij alles wat dat niet is (ook None en tekst): append() mag nooit gooien."""
    if isinstance(x, (str, bytes, bytearray)):
        return None  # float("1.5") zou slagen, maar een decoder die tekst aflevert heeft een fout
    try:
        number = float(x)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _as_float(x) -> float:
    """float(x), of NaN als dat niet kan; de aanroeper weigert NaN toch al."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


class SampleCache:
    """Ringbuffer van tijdgestempelde rijen (een waarde per signaal), strikt oplopend in tijd.

    Het geheugen ligt bij de constructie vast: een stroom die sneller of langer binnenkomt dan
    verwacht overschrijft de oudste rijen en laat niets aangroeien.

    Een schrijver en meerdere lezers tegelijk is veilig. Elk uitgelezen venster komt uit een enkel
    moment van de ring, dus de waarden in een rij horen altijd bij de tijd van die rij.
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
            raise ValueError("signals moet een lijst met namen zijn, geen losse tekst")
        names = tuple(signals)
        if not names:
            raise ValueError("signals mag niet leeg zijn")
        if not all(isinstance(name, str) for name in names):
            raise ValueError("signals mag alleen tekstnamen bevatten")
        doubles = sorted({name for name in names if names.count(name) > 1})
        if doubles:
            raise ValueError(f"signals moeten uniek zijn, dubbel: {', '.join(doubles)}")

        retention = _as_float(retention_s)
        if not (math.isfinite(retention) and retention > 0):
            raise ValueError(f"retention_s moet een eindig getal groter dan 0 zijn, kreeg {retention_s!r}")
        try:
            capacity = operator.index(max_samples)
        except TypeError:
            raise ValueError(f"max_samples moet een geheel getal zijn, kreeg {max_samples!r}") from None
        if capacity < 2:
            raise ValueError(f"max_samples moet minstens 2 zijn, kreeg {capacity}")
        step_back = _as_float(clock_step_back_s)
        if not step_back > 0:  # 'not' vangt ook NaN: een drempel die nooit waar wordt zet het klokherstel stil
            raise ValueError(f"clock_step_back_s moet groter dan 0 zijn, kreeg {clock_step_back_s!r}")
        if isinstance(flags, (str, bytes)):
            raise ValueError("flags moet een verzameling namen zijn, geen losse tekst")
        unknown = sorted(str(flag) for flag in set(flags) - set(names))
        if unknown:
            raise ValueError(f"flags bevat namen die niet in signals staan: {', '.join(unknown)}")
        if not callable(clock):
            raise TypeError("clock moet aanroepbaar zijn en epoch-seconden geven")

        self._signals = names
        self._columns = list(names)
        self._width = len(names)
        self._capacity = capacity
        self._retention_s = retention
        self._step_back_s = step_back
        self._clock = clock
        flagged = set(flags)
        self._flag_cols = np.array([i for i, name in enumerate(names) if name in flagged], dtype=np.intp)

        # Eenmalig gealloceerd; vanaf hier groeit er niets meer, hoeveel er ook binnenkomt.
        self._times = np.zeros(capacity)
        self._values = np.zeros((capacity, self._width))

        self._lock = threading.Lock()
        self._head = 0  # volgende schrijfpositie; de oudste rij staat op (head - count) % capacity
        self._count = 0
        self._newest: float | None = None
        self._appended = self._dropped_out_of_order = self._dropped_invalid = 0
        self._overwritten = self._clock_resets = 0

    def append(self, timestamp: float, values: Mapping[str, float]) -> bool:
        """Sla een rij op. True als hij is opgeslagen, False als hij is weggegooid; gooit nooit bij slechte data."""
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
                # Systeemklok teruggezet (NTP). Zonder wissen weigert de cache elk telegram tot de klok
                # het oude niveau weer haalt, en dat kan uren duren.
                self._reset_locked()
                self._clock_resets += 1
            self._store_locked(ts, row)
        return True

    def window(self, start: float, end: float, max_points: int) -> pd.DataFrame:
        """Rijen met start <= t <= end die binnen de retentie vallen, oplopend in tijd, als kopie.

        Zijn het er meer dan max_points, dan worden opeenvolgende buckets van k = ceil(rijen / max_points)
        rijen samengenomen: gemiddelde tijd, gemiddelde waarde, en het maximum voor de flag-kolommen
        (zodat een korte AAN-puls niet wegvalt).
        """
        try:
            limit = operator.index(max_points)
        except TypeError:
            raise ValueError(f"max_points moet een geheel getal zijn, kreeg {max_points!r}") from None
        if limit < 1:
            raise ValueError(f"max_points moet minstens 1 zijn, kreeg {limit}")
        start, end = float(start), float(end)
        if not start <= end:  # ook als een van beide NaN is
            return self._empty()

        # De klok wordt buiten het slot gelezen: een klok die zelf de cache bevraagt mag niet vastlopen.
        bound = self._clock() - self._retention_s
        low = start if start >= bound else bound  # bij een NaN-klok wordt dit NaN en blijft het venster leeg
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
        """Gooi alle rijen weg. De tellers blijven staan: die gaan over de levensduur van de cache."""
        with self._lock:
            self._reset_locked()

    def stats(self) -> CacheStats:
        """Tellers en uiterste tijden van dit ene moment; alles komt onder een slot, dus ze kloppen onderling."""
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
        """Aantal opgeslagen rijen, ongeacht de retentie (die geldt pas bij het uitlezen)."""
        with self._lock:
            return self._count

    # ---- opslaan --------------------------------------------------------------------------------------

    def _row(self, values: Mapping[str, float]) -> np.ndarray | None:
        """De waarden in kolomvolgorde, of None als er een ontbreekt of niet eindig is. Buiten het slot,
        en volledig gecontroleerd voordat er iets wordt opgeslagen: een half geldige rij komt er nooit in."""
        row = []
        try:
            for name in self._signals:
                value = _finite(values[name])
                if value is None:
                    return None
                row.append(value)
        except (KeyError, TypeError, IndexError):  # signaal ontbreekt, of values is niet eens een mapping
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

    # ---- uitlezen -------------------------------------------------------------------------------------

    def _locate(self, low: float, high: float) -> tuple[int, int]:
        """Logische posities [first, last) (0 = oudste rij) van de rijen met low <= t <= high.

        De ring is hoogstens twee aaneengesloten stukken die samen oplopend zijn, dus twee binaire
        zoekacties per grens volstaan en er hoeft niets van de ring te worden gekopieerd om te zoeken.
        """
        count = self._count
        if not count:
            return 0, 0
        oldest = (self._head - count) % self._capacity
        head_len = min(count, self._capacity - oldest)
        head, tail = self._times[oldest : oldest + head_len], self._times[: count - head_len]

        def position(x: float, side: str) -> int:
            k = int(head.searchsorted(x, side=side))
            if k == head_len and len(tail):  # x ligt voorbij het eerste stuk, dus zoek verder in het tweede
                k += int(tail.searchsorted(x, side=side))
            return k

        return position(low, "left"), position(high, "right")

    def _copy_rows(self, first: int, last: int) -> tuple[np.ndarray, np.ndarray]:
        """Kopie van de logische rijen [first, last); alleen dit stuk gebeurt onder het slot."""
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
        size = -(-rows // limit)  # plafond zonder float-afronding
        starts = np.arange(0, rows, size)
        counts = np.diff(np.append(starts, rows))  # de laatste bucket mag korter zijn

        firsts = times[starts]
        # Tijden middelen t.o.v. het begin van hun bucket: een som van epoch-seconden (~1,7e9) verliest anders cijfers.
        mean_times = firsts + np.add.reduceat(times - np.repeat(firsts, counts), starts) / counts
        out = np.add.reduceat(values, starts, axis=0) / counts[:, None]
        if len(self._flag_cols):
            out[:, self._flag_cols] = np.maximum.reduceat(values[:, self._flag_cols], starts, axis=0)
        return mean_times, out

    def _empty(self) -> pd.DataFrame:
        return self._frame(np.empty(0), np.empty((0, self._width)))

    def _frame(self, times: np.ndarray, values: np.ndarray) -> pd.DataFrame:
        # Zelf de index bouwen met een vaste eenheid: pandas 3 kiest voor een lege index een andere eenheid
        # dan voor een gevulde, en dan zou een leeg venster een ander type hebben dan een gevuld.
        seconds = np.floor(times)
        nanos = seconds.astype(np.int64) * _NS_PER_S + np.rint((times - seconds) * _NS_PER_S).astype(np.int64)
        index = pd.DatetimeIndex(nanos.astype("datetime64[ns]")).tz_localize("UTC")
        # copy=False: times en values zijn al een eigen kopie, nog een kopie door pandas is verspilling.
        return pd.DataFrame(values, index=index, columns=self._columns, copy=False)
