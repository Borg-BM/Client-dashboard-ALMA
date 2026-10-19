from collections.abc import Collection

import numpy as np
import pandas as pd

from alma.config import DEFAULT_COMPENSATING_MODES, SKIDS, CylinderGeometry

BAR_M2_TO_KN = 100.0  # 1 bar * 1 m2 = 1e5 N = 100 kN
MOTION_COLUMNS = ["tower_x", "tower_y", "vessel_x", "vessel_y"]
# Columns that are a state or mode: when downsampling the maximum is used, so a short period is preserved
# and no average of mode numbers is produced.
FLAG_COLUMNS = ("compensation_active", "system_mode")


def cylinder_force_kn(p_bore: pd.Series, p_rod: pd.Series, geo: CylinderGeometry) -> pd.Series:
    """Net cylinder force (kN) from the pressure on the bore and rod side (bar)."""
    a_bore = np.pi * geo.bore_diameter_m**2 / 4
    a_annulus = a_bore - np.pi * geo.rod_diameter_m**2 / 4
    return (p_bore * a_bore - p_rod * a_annulus) * BAR_M2_TO_KN


def cable_tension_kn(p_bore: pd.Series, p_rod: pd.Series, geo: CylinderGeometry) -> pd.Series:
    """Estimated cable tension (kN) per fall: cylinder force divided by the reeving ratio.

    Assumption: static approximation, no friction/acceleration of the reeving other than the efficiency.
    """
    return cylinder_force_kn(p_bore, p_rod, geo) / (geo.reeving_falls * geo.efficiency)


def epoch_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    """Seconds since 1970 as a float, regardless of the time resolution of the index (ns/us/ms)."""
    return ((index - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def rms(deviation: pd.Series) -> float:
    """RMS value of a deviation (already relative to the mean, so do not center it again:
    for the magnitude of a deviation, centering measures its spread and would report too little damping)."""
    return float(np.sqrt(np.mean(np.square(deviation)))) if len(deviation) else float("nan")


def dominant_frequency_hz(values: pd.Series, f_min: float = 0.05, f_max: float = 2.0) -> float | None:
    """Dominant vibration frequency via FFT; None if there is too little data."""
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
    """Mean position of tower and vessel over df: the zero point of the deviation."""
    return {c: float(df[c].mean()) for c in MOTION_COLUMNS}


def add_derived(
    df: pd.DataFrame,
    geo: CylinderGeometry,
    baseline: dict[str, float] | None = None,
    compensating: Collection[int] = DEFAULT_COMPENSATING_MODES,
) -> pd.DataFrame:
    """Add cable tension per skid, the compensation flag and deviations. The deviation is position minus the
    mean (by default the mean over df itself; pass baseline to use a different zero point).
    compensation_active is 1 in the modes of `compensating` (mode numbers from system_mode)."""
    out = df.copy()
    base = baseline or motion_baseline(df)
    for column in MOTION_COLUMNS:
        out[column] = out[column] - base[column]
    out["tower_dev"] = np.hypot(out["tower_x"], out["tower_y"])
    out["compensation_active"] = np.isin(np.rint(out["system_mode"]), list(compensating)).astype(float)
    for n in SKIDS:
        bore, rod = out[f"pressure_bore_skid{n}"], out[f"pressure_rod_skid{n}"]
        out[f"force_skid{n}_kn"] = cylinder_force_kn(bore, rod, geo)
        out[f"tension_skid{n}_kn"] = cable_tension_kn(bore, rod, geo)
    return out


def decimate(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """Reduce df to about max_points rows for the chart, without averaging out the vibration.

    Each bucket yields two points (minimum and maximum), so a rapidly oscillating signal over
    a long period stays visible as a band and does not vanish through coarse sampling (aliasing).
    The flag and mode columns (FLAG_COLUMNS) take the maximum per bucket, so a short on-period is not lost.
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
        for column in FLAG_COLUMNS:
            if column in part:
                part[column] = highs[column].to_numpy()
        parts.append(part)
    return pd.concat(parts).sort_index(kind="stable")


def overview(df: pd.DataFrame) -> dict:
    """Key figures for the chosen window; df must already have been through add_derived."""
    active = df["compensation_active"] > 0.5
    on, off = df[active], df[~active]

    rms_off = rms(off["tower_dev"]) if len(off) else None
    rms_on = rms(on["tower_dev"]) if len(on) else None
    reduction = None
    if rms_off and rms_on is not None:
        reduction = 100 * (1 - rms_on / rms_off)

    skids = {}
    for n in SKIDS:
        tension = on[f"tension_skid{n}_kn"] if len(on) else None
        skids[str(n)] = {
            "tension_mean_kn": float(tension.mean()) if tension is not None else None,
            "tension_peak_kn": float(tension.max()) if tension is not None else None,
            "tension_amplitude_kn": float((tension.max() - tension.min()) / 2) if tension is not None else None,
        }
    return {
        "samples": len(df),
        "active_fraction": float(active.mean()) if len(df) else None,
        "dominant_frequency_hz": dominant_frequency_hz(df["tower_x"]),
        "tower_rms_compensation_off_mm": rms_off,
        "tower_rms_compensation_on_mm": rms_on,
        "reduction_percent": reduction,
        "skids": skids,  # tension per skid, only over the period with compensation on
    }
