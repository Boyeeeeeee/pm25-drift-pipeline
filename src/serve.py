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
    state["bundle"] = joblib.load("models/latest.joblib")
    yield


app = FastAPI(title="Lagos PM2.5 next-day forecast", lifespan=lifespan)


class PredictIn(BaseModel):
    station: str
    pm25_today: float = Field(ge=0, le=1000)
    pm25_yesterday: float = Field(ge=0, le=1000)


def query(sql):
    # Short-lived read-only connection so the pipeline can still write between requests
    con = duckdb.connect(DB, read_only=True)
    try:
        return json.loads(con.execute(sql).df().to_json(orient="records", date_format="iso"))
    except duckdb.CatalogException:
        return []
    finally:
        con.close()


@app.get("/health")
def health():
    b = state["bundle"]
    return {"status": "ok", "model_trained_through": b["trained_through"],
            "stations": sorted(b["station_map"])}


@app.post("/predict")
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


@app.get("/forecasts/latest")
def forecasts_latest():
    return query("""SELECT location_name, forecast_for, made_from, round(pred,1) AS pred,
                    round(persist,1) AS persistence FROM predictions
                    WHERE forecast_for = (SELECT MAX(forecast_for) FROM predictions)
                    ORDER BY location_name""")


@app.get("/monitor/status")
def monitor_status():
    return {"input_drift": query("SELECT * FROM drift_log ORDER BY as_of DESC LIMIT 5"),
            "live_eval": query("SELECT * FROM live_eval ORDER BY as_of DESC LIMIT 5")}