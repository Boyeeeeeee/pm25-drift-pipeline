import duckdb
import pandas as pd
import pytest

from evaluate_live import evaluate, log_live_eval
from helpers import make_pairs

EMPTY = pd.DataFrame(columns=["location_name", "forecast_for", "made_from",
                              "pred", "persist", "model_trained_through"])
STATIONS = ["A", "B", "C", "D"]


def test_no_predictions():
    _, actual = make_pairs(STATIONS, 1)
    assert evaluate(EMPTY, actual)["status"] == "no_predictions"


def test_predictions_without_actuals():
    pred, actual = make_pairs(STATIONS, 3)
    actual = actual.assign(day=actual["day"] + pd.Timedelta(days=100))
    assert evaluate(pred, actual)["status"] == "no_actuals"


def test_27_scored_forecasts_is_not_enough():
    pred, actual = make_pairs(["A", "B", "C"], 9)  # 27 rows
    r = evaluate(pred, actual, window_days=14, min_rows=28)
    assert r["status"] == "insufficient" and r["n"] == 27


def test_28_scored_forecasts_is_evaluated():
    pred, actual = make_pairs(STATIONS, 7)  # 28 rows
    r = evaluate(pred, actual, window_days=14, min_rows=28)
    assert r["status"] == "evaluated" and r["n"] == 28


def test_alert_when_model_is_worse_than_persistence():
    pred, actual = make_pairs(STATIONS, 7, actual=20.0, pred=25.0, persist=20.0)
    r = evaluate(pred, actual)
    assert r["mae_model"] == pytest.approx(5.0)
    assert r["mae_persist"] == pytest.approx(0.0)
    assert r["alert"] is True


def test_no_alert_when_model_beats_persistence():
    pred, actual = make_pairs(STATIONS, 7, actual=20.0, pred=21.0, persist=25.0)
    assert evaluate(pred, actual)["alert"] is False


def test_a_tie_is_not_an_alert():
    # The registered rule is "worse than persistence", so equal error does not fire
    pred, actual = make_pairs(STATIONS, 7, actual=20.0, pred=23.0, persist=23.0)
    r = evaluate(pred, actual)
    assert r["mae_model"] == r["mae_persist"]
    assert r["alert"] is False


def test_only_the_trailing_window_counts():
    # 2 stations x 20 days. The first 6 days are terrible, the last 14 are perfect.
    pred, actual = make_pairs(["A", "B"], 20, actual=20.0, pred=20.0, persist=25.0)
    old = pd.to_datetime(pred["forecast_for"]) < pd.Timestamp("2026-10-07")
    pred.loc[old, "pred"] = 100.0
    r = evaluate(pred, actual, window_days=14, min_rows=28)
    assert r["status"] == "evaluated" and r["n"] == 28
    assert r["mae_model"] == pytest.approx(0.0)


def test_counts_distinct_model_versions_in_the_window():
    pred, actual = make_pairs(STATIONS, 7)
    pred.loc[pred.index[:10], "model_trained_through"] = "2026-10-07"
    assert evaluate(pred, actual)["n_models"] == 2


def test_live_eval_row_is_replaced_for_the_same_day():
    pred, actual = make_pairs(STATIONS, 7)
    r = evaluate(pred, actual)
    con = duckdb.connect(":memory:")
    log_live_eval(con, r)
    log_live_eval(con, r)
    assert con.execute("SELECT COUNT(*) FROM live_eval").fetchone()[0] == 1
