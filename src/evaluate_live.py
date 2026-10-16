import json
import sys

import duckdb
import pandas as pd

from data import load_clean


def stop(msg):
    print(msg)
    sys.exit(0)  # not an error: too early in the project to evaluate


cfg = json.load(open("config/monitor.json", encoding="utf-8"))
W, MIN_ROWS = cfg.get("gt_window_days", 14), cfg.get("gt_min_rows", 28)

con = duckdb.connect("data/pm25.duckdb")
try:
    pred = con.sql("SELECT * FROM predictions").df()
except duckdb.CatalogException:
    stop("No predictions table yet.")
if pred.empty:
    stop("No predictions logged yet.")
pred["forecast_for"] = pd.to_datetime(pred["forecast_for"])

actual = load_clean().rename(columns={"day": "forecast_for", "pm25": "actual"})
m = pred.merge(actual, on=["location_name", "forecast_for"], how="inner")
if m.empty:
    stop("No logged forecast has an actual yet.")

as_of = m["forecast_for"].max()
win = m[m["forecast_for"] > as_of - pd.Timedelta(days=W)]
if len(win) < MIN_ROWS:
    stop(f"Only {len(win)} scored forecasts in the window (need {MIN_ROWS}).")

mae_model = float((win["actual"] - win["pred"]).abs().mean())
mae_persist = float((win["actual"] - win["persist"]).abs().mean())
alert = bool(mae_model > mae_persist)
con.execute("""CREATE TABLE IF NOT EXISTS live_eval (
    as_of DATE PRIMARY KEY, n INTEGER, mae_model DOUBLE, mae_persist DOUBLE,
    alert BOOLEAN, logged_at TIMESTAMP)""")
con.execute("INSERT OR REPLACE INTO live_eval VALUES (?,?,?,?,?,now())",
            [as_of.date(), len(win), mae_model, mae_persist, alert])
con.close()
print(f"as_of={as_of.date()}  n={len(win)}  MAE model={mae_model:.2f}  "
      f"persist={mae_persist:.2f}  ALERT={alert}")