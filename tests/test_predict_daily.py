from datetime import date, datetime, timezone

import duckdb
import pytest

from helpers import make_bundle, make_clean
from predict_daily import build_forecasts, log_forecasts

TODAY = date(2026, 10, 19)


def test_forecast_is_today_plus_model_change():
    clean = make_clean([("Agege", "2026-10-17", 35.0), ("Agege", "2026-10-18", 31.2)])
    rows = build_forecasts(clean, make_bundle(delta=-2.0), TODAY)
    assert len(rows) == 1
    r = rows[0]
    assert r["location_name"] == "Agege"
    assert r["forecast_for"] == date(2026, 10, 19)
    assert r["made_from"] == date(2026, 10, 18)
    assert r["persist"] == pytest.approx(31.2)
    assert r["pred"] == pytest.approx(29.2)


def test_skips_station_not_in_training_map():
    clean = make_clean([("Nowhere", "2026-10-17", 35.0), ("Nowhere", "2026-10-18", 31.2)])
    assert build_forecasts(clean, make_bundle(), TODAY) == []


def test_skips_station_missing_the_previous_day():
    clean = make_clean([("Agege", "2026-10-16", 35.0), ("Agege", "2026-10-18", 31.2)])
    assert build_forecasts(clean, make_bundle(), TODAY) == []


def test_skips_stale_data():
    clean = make_clean([("Agege", "2026-10-13", 35.0), ("Agege", "2026-10-14", 31.2)])
    assert build_forecasts(clean, make_bundle(), TODAY) == []


def test_skips_target_that_is_already_past():
    # Latest data is 10-17, so the target is 10-18: not stale, but already over on 10-19.
    # This is the Dolphin Estate case: a forecast made after its day ended is not live.
    clean = make_clean([("Agege", "2026-10-16", 35.0), ("Agege", "2026-10-17", 31.2)])
    assert build_forecasts(clean, make_bundle(), TODAY) == []


def test_target_on_the_current_day_is_allowed():
    clean = make_clean([("Agege", "2026-10-17", 35.0), ("Agege", "2026-10-18", 31.2)])
    rows = build_forecasts(clean, make_bundle(), TODAY)
    assert rows[0]["forecast_for"] == TODAY


def test_the_target_is_never_later_than_today():
    # Data dated today is a partial day and is ignored, so the latest usable day is 10-18
    # and the forecast is for today, not for tomorrow.
    clean = make_clean([("Agege", "2026-10-17", 40.0), ("Agege", "2026-10-18", 35.0),
                        ("Agege", "2026-10-19", 31.2)])
    rows = build_forecasts(clean, make_bundle(), TODAY)
    assert rows[0]["forecast_for"] == TODAY


def test_each_station_is_judged_separately():
    clean = make_clean([("Agege", "2026-10-17", 35.0), ("Agege", "2026-10-18", 31.2),
                        ("Dolphin", "2026-10-13", 20.0), ("Dolphin", "2026-10-14", 18.0)])
    rows = build_forecasts(clean, make_bundle(), TODAY)
    assert [r["location_name"] for r in rows] == ["Agege"]


def _rows(pred):
    return [{"location_name": "Agege", "forecast_for": date(2026, 10, 19),
             "made_from": date(2026, 10, 18), "pred": pred, "persist": 31.2},
            {"location_name": "Dolphin", "forecast_for": date(2026, 10, 19),
             "made_from": date(2026, 10, 18), "pred": pred, "persist": 18.6}]


def test_logging_twice_adds_nothing_the_second_time():
    con = duckdb.connect(":memory:")
    assert log_forecasts(con, _rows(29.2), "2026-10-14") == 2
    assert log_forecasts(con, _rows(29.2), "2026-10-14") == 0
    assert con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 2


def test_first_forecast_is_never_overwritten():
    con = duckdb.connect(":memory:")
    log_forecasts(con, _rows(10.0), "2026-10-14")
    log_forecasts(con, _rows(99.0), "2026-10-21")
    preds = [p for (p,) in con.execute("SELECT pred FROM predictions").fetchall()]
    assert preds == [10.0, 10.0]
    trained = con.execute("SELECT DISTINCT model_trained_through FROM predictions").fetchall()
    assert trained == [(date(2026, 10, 14),)]


def test_a_new_day_adds_new_rows():
    con = duckdb.connect(":memory:")
    log_forecasts(con, _rows(10.0), "2026-10-14")
    later = _rows(11.0)
    for r in later:
        r["forecast_for"] = date(2026, 10, 20)
    assert log_forecasts(con, later, "2026-10-14") == 2


def test_a_partial_value_for_the_current_day_is_never_used():
    # A run late in the day can see today's partial average. The forecast must come from
    # the last complete day (10-18), not from the partial 10-19 reading.
    clean = make_clean([("Agege", "2026-10-17", 35.0), ("Agege", "2026-10-18", 31.2),
                        ("Agege", "2026-10-19", 5.0)])
    rows = build_forecasts(clean, make_bundle(delta=-2.0), TODAY)
    assert len(rows) == 1
    assert rows[0]["made_from"] == date(2026, 10, 18)
    assert rows[0]["forecast_for"] == TODAY
    assert rows[0]["persist"] == pytest.approx(31.2)


def test_made_at_is_utc_whatever_the_machine_timezone_is():
    con = duckdb.connect(":memory:")
    con.execute("SET TimeZone = 'Asia/Tokyo'")  # +9h: a local-time stamp would be ~9h off
    log_forecasts(con, _rows(10.0), "2026-10-14")
    made_at = con.execute("SELECT MIN(made_at) FROM predictions").fetchone()[0]
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    assert abs((made_at - now_utc).total_seconds()) < 120