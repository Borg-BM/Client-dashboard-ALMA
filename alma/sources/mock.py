"""Gesimuleerde ALMA-data: een toren die trilt op zijn eigen frequentie en een schip dat in de golven
beweegt. De compensatie staat periodiek aan en uit, zodat het effect ervan zichtbaar is."""

from datetime import datetime

import numpy as np
import pandas as pd

SAMPLE_RATE_HZ = 10.0
TOWER_FREQ_HZ = 0.28
CYCLE_S = 1200  # eerste helft compensatie uit, tweede helft aan
ACTIVE_DAMPING = 0.22  # resterende torentrilling als de compensatie aan staat


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
        envelope = 0.6 + 0.4 * np.sin(2 * np.pi * t / 600)  # wind/bouwfase wisselt
        phase = 2 * np.pi * TOWER_FREQ_HZ * t

        # Toren (MRU, mm): elliptische beweging op de eigen frequentie, trage drift en een vaste offset
        amp = 35 * envelope * np.where(active > 0, ACTIVE_DAMPING, 1.0)
        drift = 3 * np.sin(2 * np.pi * t / 900)
        tower_x = 4 + drift + amp * np.sin(phase) + _noise(idx, 1, 0.4)
        tower_y = -3 + 0.6 * drift + 0.7 * amp * np.sin(phase + 0.9) + _noise(idx, 2, 0.4)

        # Schip (MRU, mm): golfbeweging, onafhankelijk van de compensatie
        vessel_x = 60 * np.sin(2 * np.pi * t / 1500) + 220 * np.sin(2 * np.pi * 0.11 * t) + _noise(idx, 5, 4)
        vessel_y = -40 + 140 * np.sin(2 * np.pi * 0.09 * t + 1.3) + _noise(idx, 6, 4)

        # Kabel: de cilinder trekt alleen als de compensatie aan staat, in de pas met de torentrilling
        drive = active * envelope * np.sin(phase)
        pressure_bore = 80 + 25 * drive + _noise(idx, 3, 0.1)
        pressure_rod = 40 - 10 * drive + _noise(idx, 4, 0.1)

        return pd.DataFrame(
            {
                "compensation_active": active,
                "pressure_bore": pressure_bore,
                "pressure_rod": pressure_rod,
                "tower_x": tower_x,
                "tower_y": tower_y,
                "vessel_x": vessel_x,
                "vessel_y": vessel_y,
            },
            index=pd.to_datetime(t, unit="s", utc=True),
        )
