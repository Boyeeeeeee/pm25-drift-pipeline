import json
from datetime import datetime, timezone

import duckdb
import pandas as pd
from scipy.stats import ks_2samp

from data import load_clean


def compute_drift(ref, cur, window_days, min_rows, ks_threshold):
    """KS statistic between the training reference and the trailing window.

    Rule (registered in config/monitor.json): alert when KS >= ks_threshold on the
    trailing window_days of PM2.5, and only when the window has at least min_rows rows.
    """
    as_of = cur["day"].max()
    win = cur[cur["day"] > as_of - pd.Timedelta(days=window_days)]
    if len(win) < min_rows:
        return {"status": "insufficient", "n": len(win),
                "message": f"Only {len(win)} rows in window (need {min_rows}); not computing."}
    ks = float(ks_2samp(ref["pm25_today"], win["pm25"]).statistic)
    return {"status": "computed", "as_of": as_of, "ks": ks,
            "ref_median": float(ref["pm25_today"].median()),
            "win_median": float(win["pm25"].median()),
            "n": len(win), "alert": bool(ks >= ks_threshold)}


CREATE_DRIFT_LOG = """CREATE TABLE IF NOT EXISTS drift_log (
    as_of DATE PRIMARY KEY, ks DOUBLE, ref_median DOUBLE, win_median DOUBLE,
    n INTEGER, alert BOOLEAN, logged_at TIMESTAMP)"""


def log_drift(con, r):
    con.execute(CREATE_DRIFT_LOG)
    con.execute("INSERT OR REPLACE INTO drift_log VALUES (?,?,?,?,?,?,?)",
                [r["as_of"].date(), r["ks"], r["ref_median"], r["win_median"],
                 r["n"], r["alert"], datetime.now(timezone.utc)])


def main():
    cfg = json.load(open("config/monitor.json", encoding="utf-8"))
    ref = pd.read_csv("reference.csv", parse_dates=["day"])
    r = compute_drift(ref, load_clean(), cfg["window_days"], cfg["min_rows"],
                      cfg["ks_threshold"])
    if r["status"] != "computed":
        raise SystemExit(r["message"])

    con = duckdb.connect("data/pm25.duckdb")
    log_drift(con, r)
    con.close()
    print(f"as_of={r['as_of'].date()}  KS={r['ks']:.2f}  ref_median={r['ref_median']:.1f}  "
          f"window_median={r['win_median']:.1f}  n={r['n']}  ALERT={r['alert']}")


if __name__ == "__main__":
    main()