import duckdb
import joblib
import pandas as pd
from datetime import date

from data import load_clean

MAX_STALE_DAYS = 3

bundle = joblib.load("models/latest.joblib")
model, FEATS = bundle["model"], bundle["features"]
station_map, trained_through = bundle["station_map"], bundle["trained_through"]

clean = load_clean()
con = duckdb.connect("data/pm25.duckdb")
con.execute("""CREATE TABLE IF NOT EXISTS predictions (
    location_name VARCHAR, forecast_for DATE, made_from DATE,
    pred DOUBLE, persist DOUBLE, model_trained_through DATE, made_at TIMESTAMP,
    PRIMARY KEY (location_name, forecast_for))""")
before = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

for name, g in clean.groupby("location_name"):
    if name not in station_map:
        print(f"{name}: not in training station map, skipped")
        continue
    s = g.sort_values("day").set_index("day")["pm25"]
    t = s.index[-1]
    y = t - pd.Timedelta(days=1)
    if y not in s.index:
        print(f"{name}: no value for the day before {t.date()}, skipped")
        continue
    if (date.today() - t.date()).days > MAX_STALE_DAYS:
        print(f"{name}: latest day {t.date()} is stale, skipped")
        continue
    x = pd.DataFrame([{"pm25_today": s[t], "pm25_yesterday": s[y],
                       "station": station_map[name]}])[FEATS]
    pred = float(s[t] + model.predict(x)[0])
    con.execute("INSERT OR IGNORE INTO predictions VALUES (?,?,?,?,?,?,?,now())",
                [name, (t + pd.Timedelta(days=1)).date(), t.date(),
                 pred, float(s[t]), trained_through])

after = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
print(f"new forecasts logged: {after - before} (total {after})")
print(con.sql("SELECT location_name, forecast_for, round(pred,1) AS pred, "
              "round(persist,1) AS persist FROM predictions "
              "ORDER BY forecast_for DESC, location_name LIMIT 8"))
con.close()