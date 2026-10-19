from datetime import datetime
from zoneinfo import ZoneInfo

import duckdb
import joblib
import pandas as pd

from data import load_clean

MAX_STALE_DAYS = 3

CREATE_PREDICTIONS = """CREATE TABLE IF NOT EXISTS predictions (
    location_name VARCHAR, forecast_for DATE, made_from DATE,
    pred DOUBLE, persist DOUBLE, model_trained_through DATE, made_at TIMESTAMP,
    PRIMARY KEY (location_name, forecast_for))"""

INSERT_PREDICTION = """INSERT OR IGNORE INTO predictions
    (location_name, forecast_for, made_from, pred, persist,
     model_trained_through, made_at)
    VALUES (?,?,?,?,?,?, now())"""


def build_forecasts(clean, bundle, today, max_stale_days=MAX_STALE_DAYS, log=lambda msg: None):
    """Return one forecast dict per station that passes every guard.

    A station is skipped if it is not in the training station map, has no value
    for the day before its latest day, has data older than max_stale_days, or its
    target day is already in the past (such a forecast would not be made ahead of time).
    """
    model, feats, station_map = bundle["model"], bundle["features"], bundle["station_map"]
    rows = []
    for name, g in clean.groupby("location_name"):
        if name not in station_map:
            log(f"{name}: not in training station map, skipped")
            continue
        s = g.sort_values("day").set_index("day")["pm25"]
        t = s.index[-1]
        y = t - pd.Timedelta(days=1)
        if y not in s.index:
            log(f"{name}: no value for the day before {t.date()}, skipped")
            continue
        if (today - t.date()).days > max_stale_days:
            log(f"{name}: latest day {t.date()} is stale, skipped")
            continue
        target = (t + pd.Timedelta(days=1)).date()
        if target < today:
            log(f"{name}: forecast target {target} already past, skipped")
            continue
        x = pd.DataFrame([{"pm25_today": s[t], "pm25_yesterday": s[y],
                           "station": station_map[name]}])[feats]
        pred = float(s[t] + model.predict(x)[0])
        log(f"{name}: t={t.date()} target={target} pred={pred:.1f}")
        rows.append({"location_name": name, "forecast_for": target,
                     "made_from": t.date(), "pred": pred, "persist": float(s[t])})
    return rows


def log_forecasts(con, rows, trained_through):
    """Insert forecasts; the first forecast for a (station, day) is never overwritten.

    Returns the number of new rows.
    """
    con.execute(CREATE_PREDICTIONS)
    before = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    for r in rows:
        con.execute(INSERT_PREDICTION,
                    [r["location_name"], r["forecast_for"], r["made_from"],
                     r["pred"], r["persist"], trained_through])
    after = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    return after - before


def main():
    bundle = joblib.load("models/latest.joblib")
    clean = load_clean()
    today = datetime.now(ZoneInfo("Africa/Lagos")).date()
    rows = build_forecasts(clean, bundle, today, log=print)

    con = duckdb.connect("data/pm25.duckdb")
    new = log_forecasts(con, rows, bundle["trained_through"])
    total = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    print(f"new forecasts logged: {new} (total {total})")
    print(con.execute(
        "SELECT location_name, forecast_for, round(pred,1) AS pred, "
        "round(persist,1) AS persist FROM predictions "
        "ORDER BY forecast_for DESC, location_name LIMIT 8"
    ).df().to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()