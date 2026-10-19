import numpy as np
import pandas as pd

from alma.config import CylinderGeometry

BAR_M2_TO_KN = 100.0  # 1 bar * 1 m2 = 1e5 N = 100 kN
MOTION_COLUMNS = ["tower_x", "tower_y", "vessel_x", "vessel_y"]


def cylinder_force_kn(p_bore: pd.Series, p_rod: pd.Series, geo: CylinderGeometry) -> pd.Series:
    """Netto cilinderkracht (kN) uit de druk aan bodem- en stangzijde (bar)."""
    a_bore = np.pi * geo.bore_diameter_m**2 / 4
    a_annulus = a_bore - np.pi * geo.rod_diameter_m**2 / 4
    return (p_bore * a_bore - p_rod * a_annulus) * BAR_M2_TO_KN


def cable_tension_kn(p_bore: pd.Series, p_rod: pd.Series, geo: CylinderGeometry) -> pd.Series:
    """Geschatte kabelspanning (kN) per kabeldeel: cilinderkracht gedeeld door takelverhouding.

    Aanname: statische benadering, geen wrijving/versnelling van de takel behalve het rendement.
    """
    return cylinder_force_kn(p_bore, p_rod, geo) / (geo.reeving_falls * geo.efficiency)


def epoch_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    """Seconden sinds 1970 als float, ongeacht de tijdresolutie van de index (ns/us/ms)."""
    return ((index - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def rms(deviation: pd.Series) -> float:
    """Effectieve waarde van een afwijking (al ten opzichte van het gemiddelde, dus niet nog eens centreren:
    bij de grootte van een afwijking meet centreren de spreiding daarvan en rapporteert het te weinig demping)."""
    return float(np.sqrt(np.mean(np.square(deviation)))) if len(deviation) else float("nan")


def dominant_frequency_hz(values: pd.Series, f_min: float = 0.05, f_max: float = 2.0) -> float | None:
    """Dominante trillingsfrequentie via FFT; None als er te weinig data is."""
    if len(values) < 64:
        return None
    dt = float(np.median(np.diff(epoch_seconds(values.index))))
    if dt <= 0:
        return None
    x = values.to_numpy(dtype=float)
    x = (x - x.mean()) * np.hanning(len(x))
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), dt)
    band = (freqs >= f_min) & (freqs <= f_max)
    if not band.any():
        return None
    return float(freqs[band][np.argmax(spectrum[band])])


def motion_baseline(df: pd.DataFrame) -> dict[str, float]:
    """Gemiddelde positie van toren en schip over df: het nulpunt van de afwijking."""
    return {c: float(df[c].mean()) for c in MOTION_COLUMNS}


def add_derived(df: pd.DataFrame, geo: CylinderGeometry, baseline: dict[str, float] | None = None) -> pd.DataFrame:
    """Voeg kabelspanning en afwijkingen toe. De afwijking is positie minus het gemiddelde
    (standaard het gemiddelde over df zelf; geef baseline mee om een ander nulpunt te gebruiken)."""
    out = df.copy()
    base = baseline or motion_baseline(df)
    for column in MOTION_COLUMNS:
        out[column] = out[column] - base[column]
    out["tower_dev"] = np.hypot(out["tower_x"], out["tower_y"])
    out["force_kn"] = cylinder_force_kn(out["pressure_bore"], out["pressure_rod"], geo)
    out["tension_kn"] = cable_tension_kn(out["pressure_bore"], out["pressure_rod"], geo)
    return out


def decimate(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """Beperk df tot ongeveer max_points rijen voor de grafiek, zonder de trilling weg te middelen.

    Elke bucket geeft twee punten (minimum en maximum), zodat een snel oscillerend signaal over
    een lange periode als een band zichtbaar blijft en niet door grof samplen verdwijnt (aliasing).
    De compensatie-vlag neemt per bucket het maximum, dus een korte aan-periode gaat niet verloren.
    """
    if len(df) <= max_points:
        return df
    buckets = max(max_points // 2, 1)
    size = int(np.ceil(len(df) / buckets))
    group = np.arange(len(df)) // size
    grouped = df.groupby(group)
    lows, highs = grouped.min(), grouped.max()
    starts = df.index[::size]
    mids = df.index[np.minimum(np.arange(len(starts)) * size + size // 2, len(df) - 1)]

    parts = []
    for stamps, values in ((starts, lows), (mids, highs)):
        part = values.copy()
        part.index = stamps
        part["compensation_active"] = highs["compensation_active"].to_numpy()
        parts.append(part)
    return pd.concat(parts).sort_index(kind="stable")


def overview(df: pd.DataFrame) -> dict:
    """Kerncijfers over het gekozen venster; df moet add_derived al doorlopen hebben."""
    active = df["compensation_active"] > 0.5
    on, off = df[active], df[~active]

    rms_off = rms(off["tower_dev"]) if len(off) else None
    rms_on = rms(on["tower_dev"]) if len(on) else None
    reduction = None
    if rms_off and rms_on is not None:
        reduction = 100 * (1 - rms_on / rms_off)

    tension = on["tension_kn"] if len(on) else None
    return {
        "samples": len(df),
        "active_fraction": float(active.mean()) if len(df) else None,
        "dominant_frequency_hz": dominant_frequency_hz(df["tower_x"]),
        "tower_rms_compensation_off_mm": rms_off,
        "tower_rms_compensation_on_mm": rms_on,
        "reduction_percent": reduction,
        "tension_mean_kn": float(tension.mean()) if tension is not None else None,
        "tension_peak_kn": float(tension.max()) if tension is not None else None,
        "tension_amplitude_kn": float((tension.max() - tension.min()) / 2) if tension is not None else None,
    }
