from datetime import date

import duckdb
import pytest

from helpers import make_pairs
from logstore import export_logs, forecasts_for, import_logs
from evaluate_live import evaluate, log_live_eval
from predict_daily import log_forecasts


def _rows():
    return [{"location_name": "Agege", "forecast_for": date(2026, 10, 19),
             "made_from": date(2026, 10, 18), "pred": 30.7, "persist": 31.2},
            {"location_name": "Dolphin", "forecast_for": date(2026, 10, 19),
             "made_from": date(2026, 10, 18), "pred": 15.14159, "persist": 18.6}]


def _populated():
    con = duckdb.connect(":memory:")
    log_forecasts(con, _rows(), "2026-10-14")
    return con


def test_round_trip_preserves_every_value_including_made_at(tmp_path):
    src = _populated()
    export_logs(src, tmp_path)
    dst = duckdb.connect(":memory:")
    import_logs(dst, tmp_path)
    q = "SELECT * FROM predictions ORDER BY location_name"
    assert dst.execute(q).fetchall() == src.execute(q).fetchall()


def test_export_is_deterministic_so_git_diffs_stay_small(tmp_path):
    con = _populated()
    export_logs(con, tmp_path)
    first = (tmp_path / "predictions.csv").read_bytes()
    export_logs(con, tmp_path)
    assert (tmp_path / "predictions.csv").read_bytes() == first


def test_export_skips_tables_that_do_not_exist_yet(tmp_path):
    counts = export_logs(duckdb.connect(":memory:"), tmp_path)
    assert counts == {} and list(tmp_path.glob("*.csv")) == []


def test_import_with_no_csv_files_creates_empty_tables(tmp_path):
    con = duckdb.connect(":memory:")
    assert import_logs(con, tmp_path) == {"predictions": 0, "drift_log": 0, "live_eval": 0}


def test_import_never_overwrites_a_row_that_already_exists(tmp_path):
    export_logs(_populated(), tmp_path)           # CSV holds pred 30.7 for Agege
    con = duckdb.connect(":memory:")
    log_forecasts(con, [{**_rows()[0], "pred": 99.0}], "2026-10-21")
    import_logs(con, tmp_path)
    assert con.execute("SELECT pred FROM predictions WHERE location_name='Agege'").fetchone()[0] == 99.0


def test_a_restored_forecast_is_still_the_first_one_kept(tmp_path):
    # The cron path: import yesterday's log, run again today, same-day re-run must add nothing
    export_logs(_populated(), tmp_path)
    con = duckdb.connect(":memory:")
    import_logs(con, tmp_path)
    assert log_forecasts(con, [{**r, "pred": 1.0} for r in _rows()], "2026-10-21") == 0
    assert con.execute("SELECT MIN(pred) FROM predictions").fetchone()[0] > 10


def test_live_eval_round_trips_with_booleans_and_n_models(tmp_path):
    pred, actual = make_pairs(["A", "B", "C", "D"], 7)
    con = duckdb.connect(":memory:")
    log_live_eval(con, evaluate(pred, actual))
    export_logs(con, tmp_path)
    dst = duckdb.connect(":memory:")
    import_logs(dst, tmp_path)
    q = "SELECT as_of, n, mae_model, alert, n_models FROM live_eval"
    assert dst.execute(q).fetchall() == con.execute(q).fetchall()


def test_forecasts_for_counts_only_that_day():
    con = _populated()
    assert forecasts_for(con, date(2026, 10, 19)) == 2
    assert forecasts_for(con, date(2026, 10, 20)) == 0