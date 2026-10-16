import json
from datetime import datetime, timezone

import duckdb
import pandas as pd
from scipy.stats import ks_2samp

from data import load_clean

cfg = json.load(open("config/monitor.json", encoding="utf-8"))
ref = pd.read_csv("reference.csv", parse_dates=["day"])
cur = load_clean()

as_of = cur["day"].max()
win = cur[cur["day"] > as_of - pd.Timedelta(days=cfg["window_days"])]
if len(win) < cfg["min_rows"]:
    raise SystemExit(f"Only {len(win)} rows in window (need {cfg['min_rows']}); not computing.")

ks = ks_2samp(ref["pm25_today"], win["pm25"]).statistic
alert = bool(ks >= cfg["ks_threshold"])
row = (as_of.date(), float(ks), float(ref["pm25_today"].median()),
       float(win["pm25"].median()), len(win), alert, datetime.now(timezone.utc))

con = duckdb.connect("data/pm25.duckdb")
con.execute("""CREATE TABLE IF NOT EXISTS drift_log (
    as_of DATE PRIMARY KEY, ks DOUBLE, ref_median DOUBLE, win_median DOUBLE,
    n INTEGER, alert BOOLEAN, logged_at TIMESTAMP)""")
con.execute("INSERT OR REPLACE INTO drift_log VALUES (?,?,?,?,?,?,?)", row)
con.close()

print(f"as_of={as_of.date()}  KS={ks:.2f}  ref_median={row[2]:.1f}  "
      f"window_median={row[3]:.1f}  n={len(win)}  ALERT={alert}")