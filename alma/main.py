import logging
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from alma import analysis
from alma.auth import install_auth
from alma.config import ROOT, Settings, get_settings
from alma.sources import create_source
from alma.sources.base import SourceUnavailable

MAX_WINDOW_MINUTES = 60  # bovengrens van ?minutes= bij alle endpoints
MAX_ANALYSIS_POINTS = 40_000
MAX_CHART_POINTS = 2_000
LIVE_SECONDS = 10  # staart van de joystick-cirkels
BASELINE_TTL_S = 15  # hoe lang het nulpunt van de afwijking gecached blijft
UNAVAILABLE_LOG_EVERY_S = 60  # dezelfde bronfout hooguit zo vaak loggen

log = logging.getLogger("alma")


def _configure_logging() -> None:
    """Uvicorn configureert alleen zijn eigen loggers; zorg dat alma.* zichtbaar is in `docker logs`."""
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)


def _add_security_headers(app: FastAPI, settings: Settings) -> None:
    headers = {
        "Content-Security-Policy": "default-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "same-origin",
        "Cache-Control": "no-store",  # alles hier is gebruikersgebonden of live data
    }
    if settings.public_is_https:
        headers["Strict-Transport-Security"] = "max-age=31536000"

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        for name, value in headers.items():
            response.headers.setdefault(name, value)
        return response


def create_app(settings: Settings | None = None) -> FastAPI:
    _configure_logging()
    settings = settings or get_settings()
    source = create_source(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Een live bron (UDP) opent hier zijn socket: mislukt dat (poort bezet), dan start de app niet.
        start = getattr(source, "start", None)
        if callable(start):
            start()
        try:
            yield
        finally:
            stop = getattr(source, "stop", None)
            if callable(stop):
                stop()

    # Geen /docs of /openapi.json: Swagger UI laadt scripts van een CDN en past niet bij onze CSP.
    app = FastAPI(
        title="ALMA BM010 dashboard", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
    )

    def load(minutes: int):
        """Venster met afwijkingen t.o.v. het gemiddelde over dat venster."""
        end = datetime.now(timezone.utc)
        raw = source.read(end - timedelta(minutes=minutes), end, MAX_ANALYSIS_POINTS)
        return analysis.add_derived(raw, settings.cylinder)

    baselines: dict[int, tuple[float, dict | None]] = {}
    baselines_lock = threading.Lock()

    def baseline_for(minutes: int) -> dict | None:
        """Nulpunt van de afwijking voor het gekozen venster. Kort gecached, zodat de joystick die
        elke seconde ververst niet steeds het hele venster bij de bron ophaalt."""
        now = time.monotonic()
        with baselines_lock:
            cached = baselines.get(minutes)
            if cached and now - cached[0] < BASELINE_TTL_S:
                return cached[1]
        end = datetime.now(timezone.utc)
        raw = source.read(end - timedelta(minutes=minutes), end, MAX_ANALYSIS_POINTS)
        value = analysis.motion_baseline(raw) if len(raw) else None
        with baselines_lock:
            baselines[minutes] = (now, value)
        return value

    last_logged: dict = {"message": None, "at": 0.0}

    @app.exception_handler(SourceUnavailable)
    async def source_unavailable(request: Request, exc: SourceUnavailable):
        # De pagina ververst elke seconde: log dezelfde fout niet bij elke aanvraag opnieuw.
        now = time.monotonic()
        message = str(exc)
        if message != last_logged["message"] or now - last_logged["at"] > UNAVAILABLE_LOG_EVERY_S:
            log.warning("Databron niet beschikbaar: %s", message)
            last_logged.update(message=message, at=now)
        return JSONResponse({"detail": "Databron niet beschikbaar"}, status_code=503)

    @app.get("/healthz")
    def healthz():
        """Publiek, voor de Docker-healthcheck; geeft bewust niets over de configuratie weg."""
        return {"status": "ok"}

    @app.get("/api/health")
    def health():
        # retention_minutes: hoe ver terug de bron data heeft; de frontend biedt geen langere perioden aan.
        body = {
            "status": "ok",
            "source": source.name,
            "retention_minutes": min(MAX_WINDOW_MINUTES, int(getattr(source, "retention_minutes", MAX_WINDOW_MINUTES))),
        }
        stats = getattr(source, "stats", None)
        if callable(stats):  # bronnen met een live ingest (UDP) melden hier hun tellers
            body["ingest"] = stats()
        return body

    @app.get("/api/overview")
    def overview(minutes: int = Query(10, ge=1, le=MAX_WINDOW_MINUTES)):
        return analysis.overview(load(minutes))

    @app.get("/api/timeseries")
    def timeseries(minutes: int = Query(10, ge=1, le=MAX_WINDOW_MINUTES)):
        df = analysis.decimate(load(minutes), MAX_CHART_POINTS)
        return {
            "t": analysis.epoch_seconds(df.index).round(3).tolist(),
            "compensation_active": df["compensation_active"].round().astype(int).tolist(),
            "tension_kn": df["tension_kn"].round(2).tolist(),
            "tower_x": df["tower_x"].round(2).tolist(),
            "tower_y": df["tower_y"].round(2).tolist(),
            "vessel_x": df["vessel_x"].round(2).tolist(),
            "vessel_y": df["vessel_y"].round(2).tolist(),
        }

    @app.get("/api/live")
    def live(minutes: int = Query(10, ge=1, le=MAX_WINDOW_MINUTES)):
        """Actuele stand voor de statusbalk en de joystick-cirkels: compensatie aan/uit, de laatste
        afwijking van toren en schip, en een korte staart van de afgelopen seconden."""
        end = datetime.now(timezone.utc)
        recent = source.read(end - timedelta(seconds=LIVE_SECONDS), end, 2_000)
        baseline = baseline_for(minutes)
        if recent.empty or baseline is None:
            return {"ready": False}
        df = analysis.add_derived(recent, settings.cylinder, baseline=baseline)
        age_s = (pd.Timestamp(end) - recent.index[-1]).total_seconds()

        def motion(prefix: str, fullscale: float) -> dict:
            x, y = df[f"{prefix}_x"], df[f"{prefix}_y"]
            return {
                "x": round(float(x.iloc[-1]), 2),
                "y": round(float(y.iloc[-1]), 2),
                "fullscale": fullscale,
                "trail": [[round(float(a), 2), round(float(b), 2)] for a, b in zip(x, y)],
            }

        return {
            "ready": True,
            "stale": age_s > settings.stale_after_s,
            "age_s": round(age_s, 1),
            "compensation_active": bool(df["compensation_active"].iloc[-1] > 0.5),
            "tower": motion("tower", settings.tower_fullscale_mm),
            "vessel": motion("vessel", settings.vessel_fullscale_mm),
        }

    install_auth(app, settings)
    _add_security_headers(app, settings)

    # Als laatste: vangt alles op wat hierboven niet matcht. De auth-poort dekt ook deze bestanden.
    app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
    log.info(
        "ALMA dashboard gestart (bron=%s, inloggen=%s, publieke URL=%s)",
        source.name,
        "aan" if settings.auth_enabled else "UIT",
        urlsplit(settings.public_url).netloc,
    )
    return app


def app_factory() -> FastAPI:
    return create_app()
