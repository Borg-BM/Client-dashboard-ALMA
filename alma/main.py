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
from alma.config import ROOT, SKIDS, Settings, get_settings
from alma.sources import create_source
from alma.sources.base import SourceUnavailable

MAX_WINDOW_MINUTES = 60  # upper bound of ?minutes= on all endpoints
MIN_WINDOW_MINUTES = 0.25  # lower bound (15 s): a tower oscillation lasts ~6 s, so anything shorter says nothing
MAX_ANALYSIS_POINTS = 40_000
MAX_CHART_POINTS = 2_000
LIVE_SECONDS = 10  # trail of the joystick circles
BASELINE_TTL_S = 15  # how long the zero point of the deviation stays cached
UNAVAILABLE_LOG_EVERY_S = 60  # log the same source error at most this often

log = logging.getLogger("alma")


def _configure_logging() -> None:
    """Uvicorn only configures its own loggers; make sure alma.* is visible in `docker logs`."""
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
        "Cache-Control": "no-store",  # everything here is user-specific or live data
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
        # A live source (UDP) opens its socket here: if that fails (port in use), the app does not start.
        start = getattr(source, "start", None)
        if callable(start):
            start()
        try:
            yield
        finally:
            stop = getattr(source, "stop", None)
            if callable(stop):
                stop()

    # No /docs or /openapi.json: Swagger UI loads scripts from a CDN and does not fit our CSP.
    app = FastAPI(
        title="ALMA BM010 dashboard", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
    )

    compensating = settings.compensating_modes

    def load(minutes: float):
        """Window with deviations relative to the mean over that window."""
        end = datetime.now(timezone.utc)
        raw = source.read(end - timedelta(minutes=minutes), end, MAX_ANALYSIS_POINTS)
        return analysis.add_derived(raw, settings.cylinder, compensating=compensating)

    baselines: dict[float, tuple[float, dict | None]] = {}
    baselines_lock = threading.Lock()

    def baseline_for(minutes: float) -> dict | None:
        """Zero point of the deviation for the chosen window. Cached briefly, so the joystick, which
        refreshes every second, does not fetch the whole window from the source every time."""
        now = time.monotonic()
        with baselines_lock:
            cached = baselines.get(minutes)
            if cached and now - cached[0] < BASELINE_TTL_S:
                return cached[1]
            for stale in [k for k, (at, _) in baselines.items() if now - at >= BASELINE_TTL_S]:
                del baselines[stale]  # the window is a float: cleaning up old keys keeps the cache small
        end = datetime.now(timezone.utc)
        raw = source.read(end - timedelta(minutes=minutes), end, MAX_ANALYSIS_POINTS)
        value = analysis.motion_baseline(raw) if len(raw) else None
        with baselines_lock:
            baselines[minutes] = (now, value)
        return value

    def mode_of(value: float) -> dict:
        code = int(round(float(value)))
        return {"code": code, "name": settings.system_modes.get(code, f"Mode {code}")}

    last_logged: dict = {"message": None, "at": 0.0}

    @app.exception_handler(SourceUnavailable)
    async def source_unavailable(request: Request, exc: SourceUnavailable):
        # The page refreshes every second: do not log the same error again on every request.
        now = time.monotonic()
        message = str(exc)
        if message != last_logged["message"] or now - last_logged["at"] > UNAVAILABLE_LOG_EVERY_S:
            log.warning("Data source unavailable: %s", message)
            last_logged.update(message=message, at=now)
        return JSONResponse({"detail": "Data source unavailable"}, status_code=503)

    @app.get("/healthz")
    def healthz():
        """Public, for the Docker healthcheck; deliberately reveals nothing about the configuration."""
        return {"status": "ok"}

    @app.get("/api/health")
    def health():
        # retention_minutes: how far back the source has data; the frontend offers no longer periods.
        body = {
            "status": "ok",
            "source": source.name,
            "retention_minutes": min(MAX_WINDOW_MINUTES, int(getattr(source, "retention_minutes", MAX_WINDOW_MINUTES))),
        }
        stats = getattr(source, "stats", None)
        if callable(stats):  # sources with a live ingest (UDP) report their counters here
            body["ingest"] = stats()
        return body

    @app.get("/api/overview")
    def overview(minutes: float = Query(10, ge=MIN_WINDOW_MINUTES, le=MAX_WINDOW_MINUTES)):
        return analysis.overview(load(minutes))

    @app.get("/api/timeseries")
    def timeseries(minutes: float = Query(10, ge=MIN_WINDOW_MINUTES, le=MAX_WINDOW_MINUTES)):
        df = analysis.decimate(load(minutes), MAX_CHART_POINTS)
        return {
            "t": analysis.epoch_seconds(df.index).round(3).tolist(),
            "system_mode": df["system_mode"].round().astype(int).tolist(),
            "compensation_active": df["compensation_active"].round().astype(int).tolist(),
            **{f"tension_skid{n}_kn": df[f"tension_skid{n}_kn"].round(2).tolist() for n in SKIDS},
            "tower_x": df["tower_x"].round(2).tolist(),
            "tower_y": df["tower_y"].round(2).tolist(),
            "vessel_x": df["vessel_x"].round(2).tolist(),
            "vessel_y": df["vessel_y"].round(2).tolist(),
        }

    @app.get("/api/live")
    def live(minutes: float = Query(10, ge=MIN_WINDOW_MINUTES, le=MAX_WINDOW_MINUTES)):
        """Current state for the status bar and the joystick circles: the system mode, compensation on/off, the
        latest deviation of tower and vessel, and a short trail of the past seconds."""
        end = datetime.now(timezone.utc)
        recent = source.read(end - timedelta(seconds=LIVE_SECONDS), end, 2_000)
        baseline = baseline_for(minutes)
        if recent.empty or baseline is None:
            return {"ready": False}
        df = analysis.add_derived(recent, settings.cylinder, baseline=baseline, compensating=compensating)
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
            "mode": mode_of(df["system_mode"].iloc[-1]),
            "tower": motion("tower", settings.tower_fullscale_mm),
            "vessel": motion("vessel", settings.vessel_fullscale_mm),
        }

    install_auth(app, settings)
    _add_security_headers(app, settings)

    # Last: catches everything that does not match above. The auth gate also covers these files.
    app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
    log.info(
        "ALMA dashboard started (source=%s, login=%s, public URL=%s)",
        source.name,
        "on" if settings.auth_enabled else "OFF",
        urlsplit(settings.public_url).netloc,
    )
    return app


def app_factory() -> FastAPI:
    return create_app()
