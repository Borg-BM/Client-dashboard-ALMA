from datetime import datetime, timedelta, timezone

import numpy as np
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles

from alma import analysis
from alma.config import ROOT, Settings, get_settings
from alma.sources import create_source

MAX_ANALYSIS_POINTS = 40_000
MAX_CHART_POINTS = 2_000


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    source = create_source(settings)
    app = FastAPI(title="ALMA BM010 dashboard")

    def load(minutes: int):
        end = datetime.now(timezone.utc)
        raw = source.read(end - timedelta(minutes=minutes), end, MAX_ANALYSIS_POINTS)
        return analysis.add_derived(raw, settings.cylinder)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "source": source.name}

    @app.get("/api/overview")
    def overview(minutes: int = Query(10, ge=1, le=60)):
        return analysis.overview(load(minutes))

    @app.get("/api/timeseries")
    def timeseries(minutes: int = Query(10, ge=1, le=60)):
        df = load(minutes)
        df = df.iloc[:: int(np.ceil(len(df) / MAX_CHART_POINTS)) or 1]
        return {
            "t": (df.index.astype("int64") / 1e9).round(3).tolist(),
            "accel_mag": df["accel_mag"].round(4).tolist(),
            "tension_kn": df["tension_kn"].round(2).tolist(),
            "system_active": df["system_active"].round().astype(int).tolist(),
        }

    app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
    return app


def app_factory() -> FastAPI:
    return create_app()
