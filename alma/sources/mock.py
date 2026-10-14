"""Gesimuleerde ALMA-data: toren die trilt op zijn eigen frequentie, met het systeem
periodiek aan en uit zodat het effect van de compensatie zichtbaar is."""

from datetime import datetime

import numpy as np
import pandas as pd

SAMPLE_RATE_HZ = 10.0
TOWER_FREQ_HZ = 0.28
CYCLE_S = 1200  # eerste helft systeem uit, tweede helft aan
ACTIVE_DAMPING = 0.25  # resterende amplitude als het systeem aan staat


def _noise(idx: np.ndarray, salt: int, scale: float) -> np.ndarray:
    """Deterministische ruis per sample, zodat herhaalde aanvragen dezelfde data geven."""


    def uniform(k: int) -> np.ndarray:  # splitmix64-hash van (sample, salt, k) naar (0, 1)
        z = idx.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(salt * 7919 + k)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        return (z >> np.uint64(11)).astype(np.float64) / 2.0**53 + 2.0**-54

    return scale * np.sqrt(-2 * np.log(uniform(1))) * np.cos(2 * np.pi * uniform(2))  # Box-Muller


class MockSource:
    name = "mock"

    def read(self, start: datetime, end: datetime, max_points: int) -> pd.DataFrame:
        first = int(np.ceil(start.timestamp() * SAMPLE_RATE_HZ))
        last = int(np.floor(end.timestamp() * SAMPLE_RATE_HZ))
        idx = np.arange(first, last + 1)
        if len(idx) > max_points:
            idx = idx[:: int(np.ceil(len(idx) / max_points))]
        t = idx / SAMPLE_RATE_HZ

        active = ((t % CYCLE_S) >= CYCLE_S / 2).astype(float)
        envelope = 0.5 + 0.3 * np.sin(2 * np.pi * t / 600)  # wind/bouwfase wisselt
        amp = envelope * np.where(active > 0, ACTIVE_DAMPING, 1.0)
        phase = 2 * np.pi * TOWER_FREQ_HZ * t

        accel_x = amp * np.sin(phase) + _noise(idx, 1, 0.02)
        accel_y = 0.6 * amp * np.sin(phase + 0.8) + _noise(idx, 2, 0.02)
        # Cilinder trekt alleen als het systeem aan is, in tegenfase met de trilling
        drive = active * envelope * np.sin(phase)
        pressure_bore = 80 + 25 * drive + _noise(idx, 3, 0.1)
        pressure_rod = 40 - 10 * drive + _noise(idx, 4, 0.1)

        return pd.DataFrame(
            {
                "accel_x": accel_x,
                "accel_y": accel_y,
                "pressure_bore": pressure_bore,
                "pressure_rod": pressure_rod,
                "system_active": active,
            },
            index=pd.to_datetime(t, unit="s", utc=True),
        )
