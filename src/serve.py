import json
from contextlib import asynccontextmanager

import duckdb
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DB = "data/pm25.duckdb"
state = {}


@asynccontextmanager
async def lifespan(app):
    # Model is loaded once at startup: restart the API after retraining.
    state["bundle"] = joblib.load("models/latest.joblib")
    yield


app = FastAPI(title="Lagos PM2.5 next-day forecast", lifespan=lifespan)


class PredictIn(BaseModel):
    station: str
    pm25_today: float = Field(ge=0, le=1000)
    pm25_yesterday: float = Field(ge=0, le=1000)


class PredictOut(BaseModel):
    station: str
    pm25_next_day: float
    persistence_baseline: float
    model_trained_through: str
    note: str


class HealthOut(BaseModel):
    status: str
    model_trained_through: str
    stations: list[str]


class ForecastRow(BaseModel):
    location_name: str
    forecast_for: str
    made_from: str
    pred: float
    persistence: float


def query(sql):
    # Short-lived read-only connection so the pipeline can still write between requests
    try:
        con = duckdb.connect(DB, read_only=True)
    except duckdb.IOException:
        return []  # no database file (for example a fresh container before import)
    try:
        return json.loads(con.execute(sql).df().to_json(orient="records", date_format="iso"))
    except (duckdb.CatalogException, duckdb.BinderException):
        return []
    finally:
        con.close()


def scored_count():
    """Forecasts that already have an actual. None when this deployment has no raw data
    (for example the Docker image, which only holds the committed logs)."""
    rows = query("""SELECT COUNT(*) AS n FROM predictions p
                    JOIN clean_daily c ON c.location_name = p.location_name
                                       AND c.day = p.forecast_for""")
    return int(rows[0]["n"]) if rows else None


@app.get("/health", response_model=HealthOut)
def health():
    b = state["bundle"]
    return {"status": "ok", "model_trained_through": b["trained_through"],
            "stations": sorted(b["station_map"])}


@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn):
    b = state["bundle"]
    if body.station not in b["station_map"]:
        raise HTTPException(404, f"Unknown station. Known: {sorted(b['station_map'])}")
    x = pd.DataFrame([{"pm25_today": body.pm25_today,
                       "pm25_yesterday": body.pm25_yesterday,
                       "station": b["station_map"][body.station]}])[b["features"]]
    pred = body.pm25_today + float(b["model"].predict(x)[0])
    return {"station": body.station, "pm25_next_day": round(pred, 1),
            "persistence_baseline": body.pm25_today,
            "model_trained_through": b["trained_through"],
            "note": "On-demand calls are not logged and are not part of live monitoring."}


@app.get("/forecasts/latest", response_model=list[ForecastRow])
def forecasts_latest():
    return query("""SELECT location_name,
                           CAST(forecast_for AS VARCHAR) AS forecast_for,
                           CAST(made_from AS VARCHAR) AS made_from,
                           round(pred,1) AS pred,
                           round(persist,1) AS persistence
                    FROM predictions p
                    WHERE p.forecast_for = (SELECT MAX(forecast_for) FROM predictions)
                    ORDER BY location_name""")


@app.get("/monitor/status")
def monitor_status():
    cfg = json.load(open("config/monitor.json", encoding="utf-8"))
    need, window = cfg.get("gt_min_rows", 28), cfg.get("gt_window_days", 14)
    drift = query("SELECT * EXCLUDE (as_of), CAST(as_of AS VARCHAR) AS as_of "
                  "FROM drift_log ORDER BY as_of DESC LIMIT 5")
    live = query("SELECT * EXCLUDE (as_of), CAST(as_of AS VARCHAR) AS as_of "
                 "FROM live_eval ORDER BY as_of DESC LIMIT 5")
    scored = scored_count()
    if live:
        status = "evaluated"
    elif scored is None:
        status = "unavailable"  # cannot count scored forecasts without the raw data
    else:
        status = "insufficient_data"
    return {
        "note": "Last logged results of monitor.py and evaluate_live.py; nothing is recomputed on request.",
        "input_drift": drift,
        "live_eval": {"status": status, "scored_forecasts": scored,
                      "required": need, "window_days": window,
                      "history": live},
    }