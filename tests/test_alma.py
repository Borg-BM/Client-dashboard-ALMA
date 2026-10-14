from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from alma import analysis
from alma.config import CylinderGeometry, Settings
from alma.main import create_app
from alma.sources.mock import CYCLE_S, TOWER_FREQ_HZ, MockSource


def test_cylinder_force_and_tension():
    geo = CylinderGeometry(bore_diameter_m=0.2, rod_diameter_m=0.0, reeving_falls=2, efficiency=1.0)
    import pandas as pd

    force = analysis.cylinder_force_kn(pd.Series([100.0]), pd.Series([0.0]), geo)
    # 100 bar op 0,0314 m2 = 314 kN
    assert force.iloc[0] == pytest.approx(314.16, rel=1e-3)
    tension = analysis.cable_tension_kn(pd.Series([100.0]), pd.Series([0.0]), geo)
    assert tension.iloc[0] == pytest.approx(157.08, rel=1e-3)


def test_dominant_frequency_and_damping_from_mock():
    # Venster dat zowel een systeem-uit als systeem-aan periode bevat
    end = datetime.fromtimestamp(CYCLE_S * 1000 + CYCLE_S * 0.75, tz=timezone.utc)
    df = MockSource().read(end - timedelta(seconds=CYCLE_S / 2), end, 40_000)
    result = analysis.overview(analysis.add_derived(df, CylinderGeometry()))
    assert result["dominant_frequency_hz"] == pytest.approx(TOWER_FREQ_HZ, abs=0.02)
    assert result["rms_system_on"] < result["rms_system_off"]
    assert 50 < result["reduction_percent"] < 95


def test_mock_is_deterministic():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    a = MockSource().read(start, start + timedelta(seconds=30), 1000)
    b = MockSource().read(start, start + timedelta(seconds=30), 1000)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_api_endpoints():
    client = TestClient(create_app(Settings(source="mock")))
    assert client.get("/api/health").json() == {"status": "ok", "source": "mock"}
    overview = client.get("/api/overview?minutes=5").json()
    assert overview["samples"] > 0 and overview["dominant_frequency_hz"] is not None
    series = client.get("/api/timeseries?minutes=5").json()
    assert len(series["t"]) == len(series["accel_mag"]) == len(series["tension_kn"]) <= 2000
    assert client.get("/api/overview?minutes=0").status_code == 422
    assert client.get("/").status_code == 200
