"""Tests van de ibaHD-bron en `alma.ibahd_check` tegen een nagebootste server (zie fake_ibahd.py).

Dit toetst de client tegen onze lezing van de iba-documentatie; een echte server kan alleen
`python -m alma.ibahd_check` beoordelen.
"""

import json
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ibahd_api")  # optionele afhankelijkheid (extra 'ibahd'); zonder die slaan deze tests over

from fake_ibahd import CHANNELS, STORE, TEXT_CHANNEL, FakeIbaHd, analog_value, digital_state, full_id  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from alma import ibahd_check, main as main_module  # noqa: E402
from alma.config import SIGNALS, Settings  # noqa: E402
from alma.main import create_app  # noqa: E402
from alma.sources.base import SourceUnavailable  # noqa: E402
from alma.sources.ibahd import IbaHdSource, load_quick_connect, load_signal_map, request_channel_id  # noqa: E402

PLACEHOLDERS = {name: "<channel_id>" for name in SIGNALS}


@pytest.fixture(scope="module")
def fake():
    server = FakeIbaHd()
    yield server
    server.stop()


def write_files(tmp_path, server, signal_map=None, api_key=None):
    quick_connect = dict(server.quick_connect, **({"ApiKey": api_key} if api_key else {}))
    qc, sm = tmp_path / "quick_connect.json", tmp_path / "signals.json"
    qc.write_text(json.dumps(quick_connect), encoding="utf-8")
    sm.write_text(json.dumps(signal_map or {name: full_id(name) for name in SIGNALS}), encoding="utf-8")
    return qc, sm


@pytest.fixture
def files(tmp_path, fake):
    return write_files(tmp_path, fake)


def window(seconds):
    end = datetime.now(timezone.utc)
    return end - timedelta(seconds=seconds), end


def epoch_us(index):
    return ((index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(microseconds=1)).to_numpy()


# ---- de bron --------------------------------------------------------------------------------------


def test_read_returns_exactly_what_was_recorded(files):
    start, end = window(30)
    df = IbaHdSource(*files).read(start, end, 2_000)

    assert list(df.columns) == SIGNALS
    assert str(df.index.tz) == "UTC" and df.index.is_monotonic_increasing
    assert start - timedelta(seconds=1) <= df.index[0] and df.index[-1] <= end + timedelta(seconds=1)
    assert 280 <= len(df) <= 301  # 30 s op 10 Hz: bewijst dat het tijdbereik in µs goed aankomt

    t_us = epoch_us(df.index)
    for name in SIGNALS:  # float- én double-kanalen komen op de juiste tijd en waarde terug
        assert df[name].dtype == np.float64 and not df[name].isna().any()
        np.testing.assert_allclose(df[name], analog_value(name, t_us), rtol=1e-5)


def test_a_digital_channel_is_laid_on_the_time_grid_of_the_others(files):
    from alma.sources.ibahd import connect, fetch_channels, load_quick_connect

    start, end = window(30)
    channels = {"tower_x": full_id("tower_x"), "pump_running": full_id("pump_running")}
    df = fetch_channels(connect(load_quick_connect(files[0])), channels, start, end, 2_000)
    # Het digitale kanaal heeft eigen tijdstippen (alleen de wisselmomenten) en wordt op het raster gelegd
    np.testing.assert_array_equal(df["pump_running"], digital_state(epoch_us(df.index)))
    assert set(df["pump_running"]) == {0.0, 1.0}


def test_read_respects_max_points(files):
    start, end = window(60)
    df = IbaHdSource(*files).read(start, end, 100)
    assert 50 < len(df) <= 100


def test_read_without_data_gives_an_empty_frame_with_all_columns(tmp_path):
    server = FakeIbaHd(newest_age_s=300)
    try:
        df = IbaHdSource(*write_files(tmp_path, server)).read(*window(10), 2_000)
    finally:
        server.stop()
    assert df.empty and list(df.columns) == SIGNALS and str(df.index.tz) == "UTC"


def test_rows_with_gaps_are_dropped(files, monkeypatch):
    from alma.sources import ibahd

    start, end = window(10)
    clean = IbaHdSource(*files).read(start, end, 2_000)

    real = ibahd._column

    def with_gap(channel, name, index):
        values = real(channel, name, index)
        if name == "tower_x":
            values = values.copy()
            values[3] = np.nan
        return values

    monkeypatch.setattr(ibahd, "_column", with_gap)
    gappy = IbaHdSource(*files).read(start, end, 2_000)
    assert len(gappy) == len(clean) - 1 and not gappy.isna().any().any()


def test_wrong_api_key_is_reported_not_leaked(tmp_path, fake):
    source = IbaHdSource(*write_files(tmp_path, fake, api_key="not-the-key"))
    with pytest.raises(SourceUnavailable, match="UNAUTHENTICATED") as caught:
        source.read(*window(10), 100)
    assert "not-the-key" not in str(caught.value)


def test_unknown_channel_is_reported(tmp_path, fake):
    signal_map = {name: full_id(name) for name in SIGNALS} | {"tower_x": f"{STORE}\\[7:7]"}
    with pytest.raises(SourceUnavailable, match="NOT_FOUND"):
        IbaHdSource(*write_files(tmp_path, fake, signal_map)).read(*window(10), 100)


def test_channel_of_unsupported_type_is_reported(tmp_path, fake):
    signal_map = {name: full_id(name) for name in SIGNALS} | {"tower_x": f"{STORE}\\{TEXT_CHANNEL[0]}"}
    with pytest.raises(SourceUnavailable, match="STRING"):
        IbaHdSource(*write_files(tmp_path, fake, signal_map)).read(*window(10), 100)


def test_quick_connect_and_signal_map_loading(tmp_path):
    good = {"Host": "h", "Port": 1, "ApiKey": "secret-key", "TlsServerCertificate": "pem"}
    path = tmp_path / "qc.json"

    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(good).encode())  # Kladblok zet er een BOM voor
    assert load_quick_connect(path) == good

    path.write_text(json.dumps({k: v for k, v in good.items() if k != "Port"}))
    with pytest.raises(ValueError, match="Port") as caught:
        load_quick_connect(path)
    assert "secret-key" not in str(caught.value)

    path.write_text('{"ApiKey": "secret-key", oops')
    with pytest.raises(ValueError, match="geen geldige JSON") as caught:
        load_quick_connect(path)
    assert "secret-key" not in str(caught.value)

    with pytest.raises(ValueError, match="niet gevonden"):
        load_quick_connect(tmp_path / "missing.json")

    placeholders = tmp_path / "signals.json"
    placeholders.write_text(json.dumps(PLACEHOLDERS))
    with pytest.raises(ValueError, match="channel_id's"):
        load_signal_map(placeholders)


def test_request_channel_id_adds_the_store_once():
    assert request_channel_id("bm010", "[0:0]") == "bm010\\[0:0]"
    assert request_channel_id("bm010", "bm010\\[0:0]") == "bm010\\[0:0]"


# ---- het dashboard erbovenop ---------------------------------------------------------------------


def make_app(files):
    qc, sm = files
    return TestClient(create_app(Settings(_env_file=None, source="ibahd", ibahd_quick_connect=qc, signal_map=sm, auth_enabled=False)))


def test_dashboard_runs_on_ibahd_data(files):
    app = make_app(files)
    assert app.get("/api/health").json() == {"status": "ok", "source": "ibahd", "retention_minutes": 60}

    live = app.get("/api/live?minutes=1").json()
    assert live["ready"] is True and live["stale"] is False and isinstance(live["compensation_active"], bool)
    assert len(live["tower"]["trail"]) > 10

    series = app.get("/api/timeseries?minutes=1").json()
    assert len(series["t"]) > 100 and series["t"] == sorted(series["t"])
    assert app.get("/api/overview?minutes=1").json()["samples"] > 100


def test_dashboard_gives_503_when_the_server_goes_away(tmp_path):
    server = FakeIbaHd()
    try:
        app = make_app(write_files(tmp_path, server))
        assert app.get("/api/live").json()["ready"] is True
    finally:
        server.stop()
    response = app.get("/api/timeseries")
    assert response.status_code == 503 and response.json() == {"detail": "Databron niet beschikbaar"}
    assert response.headers["x-content-type-options"] == "nosniff"  # ook foutantwoorden krijgen de beveiligingsheaders
    assert app.get("/healthz").status_code == 200  # de container blijft gezond; alleen de data ontbreekt


def test_the_same_source_error_is_logged_once(monkeypatch, caplog):
    class Down:
        name = "down"

        def read(self, start, end, max_points):
            raise SourceUnavailable("ibaHD-server: UNAVAILABLE: test")

    monkeypatch.setattr(main_module, "create_source", lambda settings: Down())
    app = TestClient(create_app(Settings(_env_file=None, auth_enabled=False)))
    with caplog.at_level(logging.WARNING, logger="alma"):
        statuses = [app.get("/api/timeseries").status_code for _ in range(3)]
    assert statuses == [503, 503, 503]
    assert [r.getMessage() for r in caplog.records if "niet beschikbaar" in r.getMessage()] == [
        "Databron niet beschikbaar: ibaHD-server: UNAVAILABLE: test"
    ]


# ---- python -m alma.ibahd_check ------------------------------------------------------------------


def run_check(files, server, **kwargs):
    lines = []
    ok = ibahd_check.run(files[0], files[1], out=lines.append, **kwargs)
    text = "\n".join(lines)
    assert server.quick_connect["ApiKey"] not in text  # de ApiKey komt nooit in de uitvoer
    assert "not-the-key" not in text
    return ok, text


def test_check_passes_against_a_healthy_server(files, fake):
    ok, text = run_check(files, fake)
    assert ok, text
    for expected in (STORE, "Ingelogd", "Laatste waarde per kanaal", "Hz", "Volgende stap: zet ALMA_SOURCE=ibahd"):
        assert expected in text
    assert "FOUT" not in text and "LET OP" not in text


def test_check_with_an_empty_signal_map_still_tests_reading(tmp_path, fake):
    ok, text = run_check(write_files(tmp_path, fake, PLACEHOLDERS), fake)
    assert ok, text
    assert "nog niet ingevuld" in text and "Laatste waarde per kanaal" in text and "Meetreeks van" in text
    assert "Volgende stap: vul signals.json" in text


def test_check_suggests_the_right_id_for_a_wrong_one(tmp_path, fake):
    signal_map = {name: full_id(name) for name in SIGNALS} | {"tower_x": "[0:0]"}  # store vergeten
    ok, text = run_check(write_files(tmp_path, fake, signal_map), fake)
    assert not ok
    assert "tower_x" in text and f"Bedoelde je: {full_id('tower_x')}" in text


def test_check_normalises_schema_ids_without_the_store(tmp_path):
    server = FakeIbaHd(schema_ids="plain")
    try:
        ok, text = run_check(write_files(tmp_path, server), server)
    finally:
        server.stop()
    assert ok, text
    assert full_id("tower_x") in text  # de lijst toont het ID zoals het in signals.json en de aanroepen moet


def listed_channels(text):
    """Welke kanalen staan als rij in de kanalenlijst (andere stappen noemen ze ook, maar niet aan het begin van de regel)."""
    rows = [line.strip() for line in text.splitlines()]
    return {name for name in CHANNELS if any(row.startswith(full_id(name)) for row in rows)}


def test_check_lists_and_filters_channels(files, fake):
    _, everything = run_check(files, fake, list_all=True)
    assert listed_channels(everything) == set(CHANNELS) and TEXT_CHANNEL[1] in everything
    _, filtered = run_check(files, fake, grep="tower")
    assert listed_channels(filtered) == {"tower_x", "tower_y"}
    assert f"2 van {len(CHANNELS) + 1} kanalen" in filtered  # de kanalen van de fake plus het tekstkanaal


def test_check_reads_an_explicit_channel(files, fake):
    ok, text = run_check(files, fake, channel_ids=[full_id("pressure_bore_skid1")], minutes=0.5)
    assert ok and f"{full_id('pressure_bore_skid1')}" in text and "Meetreeks van 0.5 min" in text


def test_check_warns_when_the_newest_data_is_old(tmp_path):
    server = FakeIbaHd(newest_age_s=300)
    try:
        ok, text = run_check(write_files(tmp_path, server), server)
    finally:
        server.stop()
    assert ok, text
    assert "NO DATA" in text and "5 min" in text and "rijen" in text  # wel gelezen: venster eindigt bij de laatste meting


def test_check_flags_a_server_with_another_time_unit(tmp_path):
    server = FakeIbaHd(timestamp_scale=10)  # tijdstempels in 100 ns-ticks i.p.v. µs
    try:
        ok, text = run_check(write_files(tmp_path, server), server)
    finally:
        server.stop()
    assert not ok
    assert "geen microseconden sinds 1970" in text


def test_check_wrong_api_key(tmp_path, fake):
    ok, text = run_check(write_files(tmp_path, fake, api_key="not-the-key"), fake)
    assert not ok
    assert "UNAUTHENTICATED" in text and "ApiKey" in text
    assert "Laatste waarde" not in text  # stopt bij de eerste fout


def test_check_hostname_not_in_certificate(tmp_path):
    server = FakeIbaHd(cert_names=("other.example",))
    try:
        ok, text = run_check(write_files(tmp_path, server), server)
    finally:
        server.stop()
    assert not ok
    assert "Host 'localhost' staat niet in het certificaat" in text  # al gewaarschuwd vóór de verbinding
    assert "Waarschijnlijk de hostnaam" in text and "UNAVAILABLE" in text


def test_check_server_not_running(tmp_path):
    server = FakeIbaHd()
    files = write_files(tmp_path, server)
    server.stop()
    ok, text = run_check(files, server)
    assert not ok
    assert "Geen TCP-verbinding" in text and "Inloggen" not in text


def test_check_without_quick_connect_file(tmp_path):
    lines = []
    assert not ibahd_check.run(tmp_path / "missing.json", tmp_path / "signals.json", out=lines.append)
    assert "niet gevonden" in "\n".join(lines)


def test_check_command_line(files, capsys):
    qc, sm = files
    assert ibahd_check.main(["--quick-connect", str(qc), "--signal-map", str(sm), "--minutes", "0.2"]) == 0
    assert "Resultaat: verbinding werkt" in capsys.readouterr().out
    assert ibahd_check.main(["--quick-connect", str(qc.with_name("nope.json")), "--signal-map", str(sm)]) == 1
