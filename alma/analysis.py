import numpy as np
import pandas as pd

from alma.config import CylinderGeometry

BAR_M2_TO_KN = 100.0  # 1 bar * 1 m2 = 1e5 N = 100 kN


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


def rms(values: pd.Series) -> float:
    centered = values - values.mean()
    return float(np.sqrt(np.mean(centered**2))) if len(centered) else float("nan")


def dominant_frequency_hz(values: pd.Series, f_min: float = 0.05, f_max: float = 2.0) -> float | None:
    """Dominante trillingsfrequentie via FFT; None als er te weinig data is."""
    if len(values) < 64:
        return None
    dt = float(np.median(np.diff(values.index.view("int64"))) / 1e9)
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


def add_derived(df: pd.DataFrame, geo: CylinderGeometry) -> pd.DataFrame:
    out = df.copy()
    out["accel_mag"] = np.hypot(out["accel_x"], out["accel_y"])
    out["force_kn"] = cylinder_force_kn(out["pressure_bore"], out["pressure_rod"], geo)
    out["tension_kn"] = cable_tension_kn(out["pressure_bore"], out["pressure_rod"], geo)
    return out


def overview(df: pd.DataFrame) -> dict:
    """KPI's voor het dashboard; df moet add_derived al doorlopen hebben."""
    active = df["system_active"] > 0.5
    on, off = df[active], df[~active]

    rms_off = rms(off["accel_mag"]) if len(off) else None
    rms_on = rms(on["accel_mag"]) if len(on) else None
    reduction = None
    if rms_off and rms_on is not None:
        reduction = 100 * (1 - rms_on / rms_off)

    tension = on["tension_kn"] if len(on) else None
    return {
        "samples": len(df),
        "active_fraction": float(active.mean()) if len(df) else None,
        "dominant_frequency_hz": dominant_frequency_hz(df["accel_x"]),
        "rms_system_off": rms_off,
        "rms_system_on": rms_on,
        "reduction_percent": reduction,
        "tension_mean_kn": float(tension.mean()) if tension is not None else None,
        "tension_peak_kn": float(tension.max()) if tension is not None else None,
        "tension_amplitude_kn": float((tension.max() - tension.min()) / 2) if tension is not None else None,
    }
