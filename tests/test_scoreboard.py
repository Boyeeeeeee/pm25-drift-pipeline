import pandas as pd
import pytest

from helpers import make_pairs
from scoreboard import render, score

STATIONS = ["A", "B", "C", "D"]


def test_only_forecasts_with_an_actual_are_scored():
    pred, actual = make_pairs(["A"], 5)
    actual = actual.iloc[:3]                      # actuals exist for the first 3 days only
    m = score(pred, actual)
    assert len(m) == 3


def test_errors_are_absolute_and_use_the_right_columns():
    pred, actual = make_pairs(["A"], 1, actual=20.0, pred=25.0, persist=17.0)
    r = score(pred, actual).iloc[0]
    assert r["model_err"] == pytest.approx(5.0) and r["persist_err"] == pytest.approx(3.0)


def test_summary_reports_both_maes_and_who_was_closer():
    pred, actual = make_pairs(STATIONS, 3, actual=20.0, pred=25.0, persist=17.0)
    text = render(score(pred, actual), len(pred), 14, 28, pred, actual)
    assert "Model MAE: 5.00" in text and "Persistence MAE: 3.00" in text
    assert "persistence on 12" in text and "Model closer on 0 of 12" in text


def test_a_tie_is_reported_as_a_tie_not_a_win():
    pred, actual = make_pairs(["A"], 2, actual=20.0, pred=23.0, persist=23.0)
    text = render(score(pred, actual), len(pred), 14, 28, pred, actual)
    assert "ties 2" in text


def test_misses_are_listed_as_plainly_as_wins():
    pred, actual = make_pairs(["A"], 1, actual=40.0, pred=15.0, persist=18.0)
    text = render(score(pred, actual), len(pred), 14, 28, pred, actual)
    assert "| 15.0 | 18.0 | 40.0 | 25.0 | 22.0 | persistence |" in text


def test_too_few_forecasts_says_so_in_bold():
    pred, actual = make_pairs(STATIONS, 3)
    text = render(score(pred, actual), len(pred), 14, 28, pred, actual)
    assert "**Too few to conclude anything.**" in text and "12 so far" in text


def test_once_enough_are_scored_rule_2_is_reported():
    pred, actual = make_pairs(STATIONS, 7, actual=20.0, pred=25.0, persist=20.0)  # 28 rows, model worse
    text = render(score(pred, actual), len(pred), 14, 28, pred, actual)
    assert "ALERT=True" in text


def test_newest_forecast_day_comes_first_in_the_full_list():
    pred, actual = make_pairs(["A"], 3)
    text = render(score(pred, actual), len(pred), 14, 28, pred, actual)
    full = text.split("## Every scored forecast, newest first")[1]
    assert full.index("2026-10-03") < full.index("2026-10-02") < full.index("2026-10-01")


def test_known_late_forecasts_are_flagged():
    pred, actual = make_pairs(["A"], 1, start="2026-10-16")
    text = render(score(pred, actual), len(pred), 14, 28, pred, actual)
    assert "2026-10-16 †" in text


def test_nothing_scored_yet_gives_a_plain_placeholder():
    pred, actual = make_pairs(["A"], 2)
    actual = actual.iloc[0:0]
    text = render(score(pred, actual), len(pred), 14, 28, pred, actual)
    assert "None has an actual yet" in text and "## Summary" not in text


def test_model_versions_are_listed_separately():
    pred, actual = make_pairs(["A", "B"], 4)
    pred.loc[pred["forecast_for"] >= pd.Timestamp("2026-10-03").date(), "model_trained_through"] = "2026-10-18"
    text = render(score(pred, actual), len(pred), 14, 28, pred, actual)
    version_table = text.split("## By model version")[1].split("## Every scored")[0]
    assert "2026-10-14" in version_table and "2026-10-18" in version_table