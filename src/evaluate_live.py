import json
import sys

import duckdb
import pandas as pd

from data import load_clean

CREATE_LIVE_EVAL = """CREATE TABLE IF NOT EXISTS live_eval (
    as_of DATE PRIMARY KEY, n INTEGER, mae_model DOUBLE, mae_persist DOUBLE,
    alert BOOLEAN, n_models INTEGER, logged_at TIMESTAMP)"""


def evaluate(pred, actual, window_days=14, min_rows=28):
    """Score logged forecasts against actuals.

    Rule (registered in config/monitor.json): alert when the model's MAE over the
    trailing window is strictly worse than persistence. Nothing is evaluated until
    min_rows scored forecasts are in the window.
    """
    if pred.empty:
        return {"status": "no_predictions", "message": "No predictions logged yet."}
    pred = pred.copy()
    pred["forecast_for"] = pd.to_datetime(pred["forecast_for"])
    actual = actual.rename(columns={"day": "forecast_for", "pm25": "actual"})
    m = pred.merge(actual, on=["location_name", "forecast_for"], how="inner")
    if m.empty:
        return {"status": "no_actuals", "message": "No logged forecast has an actual yet."}

    as_of = m["forecast_for"].max()
    win = m[m["forecast_for"] > as_of - pd.Timedelta(days=window_days)]
    if len(win) < min_rows:
        return {"status": "insufficient", "n": len(win),
                "message": f"Only {len(win)} scored forecasts in the window (need {min_rows})."}

    mae_model = float((win["actual"] - win["pred"]).abs().mean())
    mae_persist = float((win["actual"] - win["persist"]).abs().mean())
    return {"status": "evaluated", "as_of": as_of, "n": len(win),
            "mae_model": mae_model, "mae_persist": mae_persist,
            "alert": bool(mae_model > mae_persist),
            "n_models": int(win["model_trained_through"].nunique())}


def log_live_eval(con, r):
    con.execute(CREATE_LIVE_EVAL)
    con.execute("INSERT OR REPLACE INTO live_eval VALUES (?,?,?,?,?,?,now())",
                [r["as_of"].date(), r["n"], r["mae_model"], r["mae_persist"],
                 r["alert"], r["n_models"]])


def main():
    cfg = json.load(open("config/monitor.json", encoding="utf-8"))
    window, min_rows = cfg.get("gt_window_days", 14), cfg.get("gt_min_rows", 28)

    con = duckdb.connect("data/pm25.duckdb")
    try:
        pred = con.sql("SELECT * FROM predictions").df()
    except duckdb.CatalogException:
        print("No predictions table yet.")
        sys.exit(0)  # not an error: too early in the project to evaluate

    r = evaluate(pred, load_clean(), window, min_rows)
    if r["status"] != "evaluated":
        print(r["message"])
        sys.exit(0)

    log_live_eval(con, r)
    con.close()
    print(f"as_of={r['as_of'].date()}  n={r['n']}  MAE model={r['mae_model']:.2f}  "
          f"persist={r['mae_persist']:.2f}  models_in_window={r['n_models']}  ALERT={r['alert']}")


if __name__ == "__main__":
    main()