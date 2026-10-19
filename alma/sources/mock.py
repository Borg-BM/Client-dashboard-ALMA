"""Gesimuleerde ALMA-data: een toren die trilt op zijn eigen frequentie en een schip dat in de golven
beweegt. Het systeem doorloopt de modi (off, HPU on, position control, CT low, damping), zodat het effect
ervan zichtbaar is. Twee skids trekken aan de toren: skid 1 in X, skid 2 in Y."""

from datetime import datetime

import numpy as np
import pandas as pd

SAMPLE_RATE_HZ = 10.0
TOWER_FREQ_HZ = 1 / 6  # een oscillatie duurt ~6 s
CYCLE_S = 1200  # per cyclus doorloopt het systeem alle modi; de tweede helft compenseert
CT_LOW_DAMPING = 0.6  # resterende torentrilling in CT low
ACTIVE_DAMPING = 0.22  # resterende torentrilling in damping

# (begin in s binnen de cyclus, modenummer); dezelfde nummers als DEFAULT_SYSTEM_MODES in alma.config
MODE_STARTS = np.array([0, 200, 350, 600, 750])
MODE_CODES = np.array([0, 1, 2, 3, 4])
OFF, HPU_ON, POSITION_CONTROL, CT_LOW, DAMPING = MODE_CODES


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

        mode = MODE_CODES[np.searchsorted(MODE_STARTS, t % CYCLE_S, side="right") - 1]
        envelope = 0.6 + 0.4 * np.sin(2 * np.pi * t / 600)  # wind/bouwfase wisselt
        phase = 2 * np.pi * TOWER_FREQ_HZ * t

        # Toren (MRU, mm): elliptische beweging op de eigen frequentie, trage drift en een vaste offset
        remaining = np.select([mode == CT_LOW, mode == DAMPING], [CT_LOW_DAMPING, ACTIVE_DAMPING], default=1.0)
        amp = 35 * envelope * remaining
        drift = 3 * np.sin(2 * np.pi * t / 900)
        tower_x = 4 + drift + amp * np.sin(phase) + _noise(idx, 1, 0.4)
        tower_y = -3 + 0.6 * drift + 0.7 * amp * np.sin(phase + 0.9) + _noise(idx, 2, 0.4)

        # Schip (MRU, mm): golfbeweging, onafhankelijk van de compensatie
        vessel_x = 60 * np.sin(2 * np.pi * t / 1500) + 220 * np.sin(2 * np.pi * 0.11 * t) + _noise(idx, 5, 4)
        vessel_y = -40 + 140 * np.sin(2 * np.pi * 0.09 * t + 1.3) + _noise(idx, 6, 4)

        # Skids: met de HPU aan staat er standaardspanning op; in CT low trekt een skid zacht, in damping
        # trekt hij in de pas met de torentrilling (skid 1 met X, skid 2 met Y)
        standby = (mode >= HPU_ON).astype(float)
        effort = np.select([mode == CT_LOW, mode == DAMPING], [0.4, 1.0], default=0.0) * envelope
        drives = {1: effort * np.sin(phase), 2: effort * np.sin(phase + 0.9)}
        pressures = {}
        for n, salt, bore, rod in ((1, 3, 60, 30), (2, 7, 52, 27)):  # skid 2 staat iets lager af
            pressures[f"pressure_bore_skid{n}"] = standby * (bore + 25 * drives[n]) + _noise(idx, salt, 0.1)
            pressures[f"pressure_rod_skid{n}"] = standby * (rod - 10 * drives[n]) + _noise(idx, salt + 1, 0.1)

        return pd.DataFrame(
            {
                "system_mode": mode.astype(float),
                **pressures,
                "tower_x": tower_x,
                "tower_y": tower_y,
                "vessel_x": vessel_x,
                "vessel_y": vessel_y,
            },
            index=pd.to_datetime(t, unit="s", utc=True),
        )
