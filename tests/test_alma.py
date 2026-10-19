from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from alma import analysis
from alma import main as main_module
from alma.config import CylinderGeometry, Settings
from alma.main import create_app
from alma.sources.mock import CYCLE_S, TOWER_FREQ_HZ, MockSource

GEO = CylinderGeometry()


def mock_window(seconds: float, ends_at: float = CYCLE_S * 1000 + CYCLE_S * 0.75) -> pd.DataFrame:
    end = datetime.fromtimestamp(ends_at, tz=timezone.utc)
    return MockSource().read(end - timedelta(seconds=seconds), end, 40_000)


def test_cylinder_force_and_tension():
    geo = CylinderGeometry(bore_diameter_m=0.2, rod_diameter_m=0.0, reeving_falls=2, efficiency=1.0)

    force = analysis.cylinder_force_kn(pd.Series([100.0]), pd.Series([0.0]), geo)
    # 100 bar op 0,0314 m2 = 314 kN
    assert force.iloc[0] == pytest.approx(314.16, rel=1e-3)
    tension = analysis.cable_tension_kn(pd.Series([100.0]), pd.Series([0.0]), geo)
    assert tension.iloc[0] == pytest.approx(157.08, rel=1e-3)


def test_deviation_is_relative_to_the_mean():
    raw = mock_window(CYCLE_S / 2)
    assert abs(raw["tower_x"].mean()) > 1  # de ruwe MRU-waarde heeft een offset...
    derived = analysis.add_derived(raw, GEO)
    for column in analysis.MOTION_COLUMNS:  # ...die na aftrekken van het gemiddelde weg is
        assert derived[column].mean() == pytest.approx(0, abs=1e-9)


def test_deviation_can_use_a_given_baseline():
    raw = mock_window(60)
    base = {c: 100.0 for c in analysis.MOTION_COLUMNS}
    derived = analysis.add_derived(raw, GEO, baseline=base)
    assert derived["tower_x"].iloc[0] == pytest.approx(raw["tower_x"].iloc[0] - 100.0)


def test_dominant_frequency_and_damping_from_mock():
    # Een volle cyclus: uit- en aan-deel bevatten dan beide een hele 'wind'-periode, dus een eerlijke vergelijking
    df = analysis.add_derived(mock_window(CYCLE_S), GEO)
    result = analysis.overview(df)
    assert result["dominant_frequency_hz"] == pytest.approx(TOWER_FREQ_HZ, abs=0.02)
    assert result["tower_rms_compensation_on_mm"] < result["tower_rms_compensation_off_mm"]
    assert 50 < result["reduction_percent"] < 95


def test_mock_is_deterministic():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    a = MockSource().read(start, start + timedelta(seconds=30), 1000)
    b = MockSource().read(start, start + timedelta(seconds=30), 1000)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_decimate_keeps_peaks_and_short_active_periods():
    n = 20_000
    index = pd.date_range("2026-01-01", periods=n, freq="100ms", tz="UTC")
    df = pd.DataFrame({c: np.zeros(n) for c in ["compensation_active", *analysis.MOTION_COLUMNS, "tension_kn"]}, index=index)
    df.loc[index[12_345], "tower_x"] = 99.0  # een enkele piek
    df.loc[index[7_777], "tower_y"] = -88.0  # en een dal
    df.loc[index[3_000:3_002], "compensation_active"] = 1.0  # en een heel korte aan-periode

    out = analysis.decimate(df, 2_000)
    assert len(out) <= 2_000 + 2
    assert out.index.is_monotonic_increasing
    assert out["tower_x"].max() == 99.0 and out["tower_y"].min() == -88.0
    assert out["compensation_active"].max() == 1.0


def test_decimate_leaves_small_frames_alone():
    df = mock_window(30)
    assert analysis.decimate(df, 2_000) is df


def test_epoch_seconds_does_not_depend_on_index_resolution():
    for unit in ("ns", "us", "ms", "s"):
        index = pd.DatetimeIndex(pd.to_datetime([1_700_000_000, 1_700_000_001], unit="s", utc=True)).as_unit(unit)
        assert analysis.epoch_seconds(index).tolist() == [1_700_000_000.0, 1_700_000_001.0]


@pytest.fixture
def client():
    return TestClient(create_app(Settings(_env_file=None, source="mock", auth_enabled=False)))


def test_api_endpoints_without_login(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/api/health").json() == {"status": "ok", "source": "mock", "retention_minutes": 60}
    overview = client.get("/api/overview?minutes=5").json()
    assert overview["samples"] > 0 and overview["dominant_frequency_hz"] is not None
    assert client.get("/api/overview?minutes=0").status_code == 422
    assert client.get("/").status_code == 200
    assert client.get("/vendor/uplot/uPlot.iife.min.js").status_code == 200
    assert client.get("/assets/bm-logo.png").headers["content-type"] == "image/png"
    assert client.get("/docs").status_code == 404  # Swagger UI uit: past niet bij de CSP


def test_timeseries_has_everything_the_charts_need(client):
    series = client.get("/api/timeseries?minutes=60").json()
    keys = ["t", "compensation_active", "tension_kn", "tower_x", "tower_y", "vessel_x", "vessel_y"]
    assert {len(series[k]) for k in keys} == {len(series["t"])}
    assert 0 < len(series["t"]) <= 2_002
    assert set(series["compensation_active"]) <= {0, 1}
    assert series["t"] == sorted(series["t"])


def test_live_gives_status_and_dots(client):
    live = client.get("/api/live?minutes=10").json()
    assert live["ready"] is True and live["stale"] is False
    assert isinstance(live["compensation_active"], bool)
    for who, fullscale in (("tower", 60.0), ("vessel", 500.0)):
        m = live[who]
        assert m["fullscale"] == fullscale
        assert len(m["trail"]) > 10 and all(len(p) == 2 for p in m["trail"])
        assert [m["x"], m["y"]] == m["trail"][-1]  # de stip staat op het laatste punt van de staart
    assert client.get("/api/live?minutes=0").status_code == 422


def test_live_status_follows_the_data(monkeypatch):
    class Stub:
        name = "stub"

        def __init__(self, df):
            self.df = df

        def read(self, start, end, max_points):
            return self.df

    # Laatste meting is 5 minuten oud: de balk moet 'NO DATA' kunnen tonen i.p.v. een oude status
    old = mock_window(30, ends_at=datetime.now(timezone.utc).timestamp() - 300)
    monkeypatch.setattr(main_module, "create_source", lambda settings: Stub(old))
    stale = TestClient(create_app(Settings(_env_file=None, auth_enabled=False))).get("/api/live").json()
    assert stale["ready"] is True and stale["stale"] is True and stale["age_s"] > 200

    # Lege bron: geen crash, gewoon 'niet klaar' en lege grafieken
    now = datetime.now(timezone.utc)
    empty = MockSource().read(now, now - timedelta(seconds=1), 10)
    assert empty.empty
    monkeypatch.setattr(main_module, "create_source", lambda settings: Stub(empty))
    app = TestClient(create_app(Settings(_env_file=None, auth_enabled=False)))
    assert app.get("/api/live").json() == {"ready": False}
    series = app.get("/api/timeseries").json()
    assert series["t"] == [] and series["tension_kn"] == []
    assert app.get("/api/overview").status_code == 200


def test_cylinder_geometry_from_environment(monkeypatch):
    monkeypatch.setenv("ALMA_CYLINDER__BORE_DIAMETER_M", "0.25")
    monkeypatch.setenv("ALMA_CYLINDER__REEVING_FALLS", "6")
    settings = Settings(_env_file=None, auth_enabled=False)
    assert settings.cylinder.bore_diameter_m == 0.25
    assert settings.cylinder.reeving_falls == 6
    assert settings.cylinder.rod_diameter_m == 0.140  # niet opgegeven: default blijft
