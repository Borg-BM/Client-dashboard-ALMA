"""De UDP-bron als geheel: echte UDP-pakketten over loopback -> ontvanger -> cache -> API.

De klok van de bron wordt door de test bediend: elk pakket krijgt een tijdstip dat de test kiest, en de
test wacht tot het pakket is verwerkt voordat het volgende de deur uit gaat. Zo is de tijdlijn in de cache
exact bekend zonder dat de test in real time hoeft te draaien.
"""

import json
import socket
import time
from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

# TIJDELIJK (verwijderen zodra ontvanger, simulator en heartbeat gebouwd zijn): tijdens het bouwen in golven
# bestaan die onderdelen nog niet, en dan zou deze test de hele suite als verzamelfout laten mislukken.
try:
    import alma.ingest.receiver  # noqa: F401
    import alma.sources.udp  # noqa: F401
    from alma.sources.mock import signals_at  # noqa: F401
except ImportError as missing:
    pytest.skip(f"onderdeel nog niet gebouwd: {missing}", allow_module_level=True)

from alma import main as main_module
from alma.config import ROOT, SIGNALS, Settings
from alma.ingest.format import PacketFormat
from alma.main import create_app
from alma.sources.mock import TOWER_FREQ_HZ, signals_at
from alma.sources.udp import UdpSource

DEFAULT = json.loads((ROOT / "udp_format.json").read_text(encoding="utf-8"))
FORMAT = PacketFormat.load(ROOT / "udp_format.json", required_signals=SIGNALS)
STEP_S = 0.05  # ibaPDA: minimale updatetijd, 20 telegrammen per seconde


def with_heartbeat(**options) -> PacketFormat:
    """Het standaardtelegram met andere hartslag-instellingen."""
    return PacketFormat.from_dict({**DEFAULT, "heartbeat": {**DEFAULT["heartbeat"], **options}})


class Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def processed(stats: dict) -> int:
    """Aantal telegrammen dat helemaal is afgehandeld, wat er ook mee gebeurde."""
    return stats["accepted"] + stats["dropped"] + stats["blocked"] + stats["malformed"]


def feed(source, clock, first, count, *, fmt=FORMAT, heartbeat=lambda tick: tick, skip=()) -> None:
    """Stuur `count` telegrammen met de gesimuleerde signalen, op tijdstip first, first+STEP_S, ...
    De tikken in `skip` worden niet verstuurd (verlies op de verbinding); `heartbeat(tick)` is de waarde
    van de hartslag in het telegram."""
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for tick in range(count):
            clock.now = first + tick * STEP_S
            if tick in skip:
                continue
            values = {name: float(v[0]) for name, v in signals_at(np.array([clock.now])).items()}
            before = processed(source.receiver.stats())
            sender.sendto(fmt.encode(values, heartbeat=heartbeat(tick)), ("127.0.0.1", source.port))
            deadline = time.monotonic() + 2
            while processed(source.receiver.stats()) == before:
                assert time.monotonic() < deadline, f"telegram {tick} is niet verwerkt"
                time.sleep(0.0002)
    finally:
        sender.close()


def make_source(clock: Clock, fmt: PacketFormat = FORMAT, **kwargs) -> UdpSource:
    options = dict(bind="127.0.0.1", port=0, allowed_sources=["127.0.0.1"], cache_minutes=10, clock=clock)
    return UdpSource(fmt, **{**options, **kwargs})


def make_app(monkeypatch, source: UdpSource) -> TestClient:
    monkeypatch.setattr(main_module, "create_source", lambda settings: source)
    settings = Settings(_env_file=None, auth_enabled=False, source="udp", udp_allow_any_source=True)
    return TestClient(create_app(settings))  # gebruik als context manager: dan start de lifespan de ontvanger


def test_dashboard_runs_on_live_udp_data(monkeypatch):
    now = time.time()
    clock = Clock(now - 60)
    source = make_source(clock)
    with make_app(monkeypatch, source) as app:
        feed(source, clock, now - 60, 1200)  # een minuut data, tot ongeveer nu

        health = app.get("/api/health").json()
        assert health["source"] == "udp" and health["retention_minutes"] == 10
        ingest = health["ingest"]
        assert ingest["packets"] == 1200 and ingest["listening"] is True and ingest["frozen"] is False
        assert 59 <= ingest["held_s"] <= 60
        assert ingest["loss_source"] == "estimate" and ingest["loss_percent"] < 1  # volledige stroom: geen verlies
        assert "127.0.0.1" not in str(health)  # klanten mogen onze netwerkadressen niet zien

        live = app.get("/api/live?minutes=1").json()
        assert live["ready"] is True and live["stale"] is False
        assert len(live["tower"]["trail"]) > 10

        series = app.get("/api/timeseries?minutes=1").json()
        assert len(series["t"]) > 500 and series["t"] == sorted(series["t"])
        assert set(series["compensation_active"]) <= {0, 1}

        overview = app.get("/api/overview?minutes=1").json()
        assert overview["samples"] >= 1100
        assert overview["dominant_frequency_hz"] == pytest.approx(TOWER_FREQ_HZ, abs=0.03)


def test_packet_loss_is_exact_when_the_heartbeat_counts_telegrams(monkeypatch):
    now = time.time()
    clock = Clock(now - 10)
    fmt = with_heartbeat(counts_telegrams=True)
    source = make_source(clock, fmt)
    with make_app(monkeypatch, source) as app:
        feed(source, clock, now - 10, 100, fmt=fmt, skip=(30, 31, 32, 70))  # een gat van 3 en een van 1
        ingest = app.get("/api/health").json()["ingest"]
        assert ingest["packets"] == 96 and ingest["loss_source"] == "heartbeat"
        assert ingest["loss_percent"] == pytest.approx(4.0)
        series = app.get("/api/timeseries?minutes=1").json()  # het gat is gewoon een gat: geen crash, geen NaN
        assert len(series["t"]) > 50 and all(isinstance(v, (int, float)) for v in series["tower_x"])


def test_packet_loss_is_estimated_without_a_counting_heartbeat(monkeypatch):
    now = time.time()
    clock = Clock(now - 15)
    source = make_source(clock)
    with make_app(monkeypatch, source) as app:
        feed(source, clock, now - 15, 300, skip=range(10, 290, 10))  # 28 van de 300 kwijt
        ingest = app.get("/api/health").json()["ingest"]
        assert ingest["packets"] == 272 and ingest["loss_source"] == "estimate"
        assert ingest["loss_percent"] == pytest.approx(100 * 28 / 300, abs=0.5)


def test_a_frozen_sender_is_withheld_and_the_dashboard_goes_stale(monkeypatch):
    """ibaPDA of de PLC blijft dezelfde waarden sturen (met verse aankomsttijden): de hartslag staat stil. Dat mag
    niet als live data op het scherm komen."""
    now = time.time()
    clock = Clock(now - 30)
    source = make_source(clock)  # standaardformaat: hartslag zonder tellen, bevroren na 5 s
    with make_app(monkeypatch, source) as app:
        feed(source, clock, now - 30, 300, heartbeat=lambda tick: min(tick, 99))  # na tik 99 staat de hartslag stil
        ingest = app.get("/api/health").json()["ingest"]
        assert ingest["frozen"] is True
        assert 198 <= ingest["packets"] <= 201  # opgeslagen tot 5 s na de laatste verandering
        newest = source.cache.stats().newest
        assert newest < now - 30 + 200 * STEP_S

        # de hartslag beweegt weer: vanaf het eerste telegram wordt er weer opgeslagen
        feed(source, clock, now - 15, 20, heartbeat=lambda tick: 300 + tick)
        ingest = app.get("/api/health").json()["ingest"]
        assert ingest["frozen"] is False and ingest["packets"] >= 198 + 20


def test_a_toggling_heartbeat_never_freezes(monkeypatch):
    now = time.time()
    clock = Clock(now - 20)
    source = make_source(clock)
    with make_app(monkeypatch, source) as app:
        feed(source, clock, now - 20, 400, heartbeat=lambda tick: tick % 2)
        ingest = app.get("/api/health").json()["ingest"]
        assert ingest["frozen"] is False and ingest["packets"] == 400


def test_status_goes_stale_when_the_stream_stops(monkeypatch):
    now = time.time()
    clock = Clock(now - 120)
    source = make_source(clock)
    with make_app(monkeypatch, source) as app:
        feed(source, clock, now - 120, 400)  # eindigt ~100 s geleden
        live = app.get("/api/live?minutes=5").json()
        assert live["ready"] is True and live["stale"] is True and live["age_s"] > 60


def test_without_packets_the_dashboard_says_not_ready(monkeypatch):
    source = make_source(Clock(time.time()))
    with make_app(monkeypatch, source) as app:
        assert app.get("/api/live").json() == {"ready": False}
        assert app.get("/api/timeseries").json()["t"] == []
        assert app.get("/api/overview").status_code == 200
        ingest = app.get("/api/health").json()["ingest"]
        assert ingest["listening"] is True and ingest["packets"] == 0 and ingest["last_packet_age_s"] is None
        assert ingest["loss_percent"] is None and ingest["loss_source"] is None


def test_the_cache_only_serves_what_it_keeps(monkeypatch):
    now = time.time()
    clock = Clock(now - 400)
    source = make_source(clock, cache_minutes=2)
    with make_app(monkeypatch, source) as app:
        feed(source, clock, now - 400, 8000)  # 400 s data, de cache houdt 120 s vast
        assert app.get("/api/health").json()["retention_minutes"] == 2
        series = app.get("/api/timeseries?minutes=10").json()
        assert series["t"][-1] - series["t"][0] <= 121


def test_app_refuses_to_start_when_the_port_is_taken(monkeypatch):
    blocker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    blocker.bind(("127.0.0.1", 0))
    try:
        source = make_source(Clock(time.time()), port=blocker.getsockname()[1])
        with pytest.raises(RuntimeError, match="UDP-poort"):
            with make_app(monkeypatch, source):
                pass
    finally:
        blocker.close()


def test_shutdown_frees_the_port(monkeypatch):
    source = make_source(Clock(time.time()))
    with make_app(monkeypatch, source):
        port = source.port
    again = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        again.bind(("127.0.0.1", port))
    finally:
        again.close()


def test_source_from_settings_applies_the_format_and_the_filter():
    def run(allowed):
        settings = Settings(
            _env_file=None, auth_enabled=False, source="udp", udp_bind="127.0.0.1", udp_port=0, udp_allowed_sources=allowed
        )
        source = UdpSource.from_settings(settings)
        source.start()
        try:
            values = {name: 1.0 for name in SIGNALS}
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                sender.sendto(FORMAT.encode(values), ("127.0.0.1", source.port))
            deadline = time.monotonic() + 2
            while source.receiver.stats()["received"] == 0 and time.monotonic() < deadline:
                time.sleep(0.01)
            return source.receiver.stats()
        finally:
            source.stop()

    assert run(["127.0.0.1"])["accepted"] == 1
    blocked = run(["10.9.9.9"])
    assert blocked["accepted"] == 0 and blocked["blocked"] == 1


def test_allow_any_source_on_a_reachable_address_needs_a_check_word(tmp_path):
    layout = {key: value for key, value in DEFAULT.items() if key != "heartbeat"}
    with_key = {**layout, "checks": [{"offset": 28, "type": "DWORD", "equals": 123456789}]}

    def settings(layout, bind):
        path = tmp_path / "format.json"
        path.write_text(json.dumps(layout), encoding="utf-8")
        return Settings(
            _env_file=None, auth_enabled=False, source="udp", udp_format=path, udp_allow_any_source=True, udp_bind=bind
        )

    with pytest.raises(ValueError, match="checks"):
        UdpSource.from_settings(settings(layout, "0.0.0.0"))
    UdpSource.from_settings(settings(layout, "127.0.0.1"))  # lokaal testen blijft kunnen
    UdpSource.from_settings(settings(with_key, "0.0.0.0"))  # met sleutelwoord mag het, ook zonder bronfilter


def test_read_returns_aware_utc_frames_with_all_signals():
    now = time.time()
    clock = Clock(now)
    source = make_source(clock)
    source.start()
    try:
        feed(source, clock, now - 5, 100)
        frame = source.read(datetime.fromtimestamp(now - 10, timezone.utc), datetime.fromtimestamp(now, timezone.utc), 10_000)
    finally:
        source.stop()
    assert list(frame.columns) == SIGNALS and str(frame.index.tz) == "UTC" and len(frame) == 100
