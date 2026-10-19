import duckdb
import pandas as pd
import pytest

from data import load_clean

RAW_AIR = "CREATE TABLE raw_openaq_daily (location_name VARCHAR, day DATE, value DOUBLE, pct_complete DOUBLE)"
RAW_WEATHER = ("CREATE TABLE raw_weather_daily (day DATE, temp_mean DOUBLE, precip_sum DOUBLE, "
               "wind_max DOUBLE, wind_dir DOUBLE)")


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "test.duckdb")
    con = duckdb.connect(path)
    con.execute(RAW_AIR)
    con.execute(RAW_WEATHER)
    con.close()
    return path


def _insert(path, rows):
    con = duckdb.connect(path)
    con.executemany("INSERT INTO raw_openaq_daily VALUES (?,?,?,?)", rows)
    con.close()


def test_coverage_filter_keeps_75_percent_and_drops_below(db):
    _insert(db, [("A", "2026-10-01", 10.0, 75.0),
                 ("A", "2026-10-02", 11.0, 74.9),
                 ("A", "2026-10-03", 12.0, 100.0)])
    clean = load_clean(db)
    assert list(clean["day"].dt.day) == [1, 3]


def test_load_clean_returns_datetime_days_and_expected_columns(db):
    _insert(db, [("A", "2026-10-01", 10.0, 100.0)])
    clean = load_clean(db)
    assert list(clean.columns) == ["location_name", "day", "pm25"]
    assert pd.api.types.is_datetime64_any_dtype(clean["day"])


def test_features_never_pair_days_across_a_gap(db):
    # Days 1, 2, 4, 5 (day 3 missing)
    _insert(db, [("A", f"2026-10-0{d}", float(d * 10), 100.0) for d in (1, 2, 4, 5)])
    con = duckdb.connect(db)
    for stmt in open("sql/views.sql", encoding="utf-8").read().split(";"):
        if stmt.strip():
            con.execute(stmt)
    feats = con.execute("SELECT day, pm25_today, pm25_yesterday, target_next_day "
                        "FROM features ORDER BY day").df()
    con.close()
    # Day 1 pairs with day 2. Day 2 has no day 3, so it is dropped.
    # Day 4 pairs with day 5 and has no yesterday (day 3 missing). Day 5 has no day 6.
    assert list(feats["pm25_today"]) == [10.0, 40.0]
    assert list(feats["target_next_day"]) == [20.0, 50.0]
    assert pd.isna(feats["pm25_yesterday"].iloc[0]) and pd.isna(feats["pm25_yesterday"].iloc[1])


def test_features_have_yesterday_when_consecutive(db):
    _insert(db, [("A", f"2026-10-0{d}", float(d * 10), 100.0) for d in (1, 2, 3)])
    con = duckdb.connect(db)
    for stmt in open("sql/views.sql", encoding="utf-8").read().split(";"):
        if stmt.strip():
            con.execute(stmt)
    feats = con.execute("SELECT pm25_today, pm25_yesterday, target_next_day "
                        "FROM features ORDER BY day").df()
    con.close()
    # Day 2 is the only row with both neighbours
    assert feats["pm25_yesterday"].iloc[1] == 10.0
    assert feats["target_next_day"].iloc[1] == 30.0


def test_stations_are_not_mixed(db):
    _insert(db, [("A", "2026-10-01", 10.0, 100.0), ("B", "2026-10-02", 99.0, 100.0)])
    con = duckdb.connect(db)
    for stmt in open("sql/views.sql", encoding="utf-8").read().split(";"):
        if stmt.strip():
            con.execute(stmt)
    n = con.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    con.close()
    assert n == 0  # A's day 2 and B's day 1 do not exist
