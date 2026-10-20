from datetime import date, datetime, timezone

import duckdb
import pytest

import ingest

NOW = datetime(2026, 10, 19, 6, 0, tzinfo=timezone.utc)


def record(local_from, utc_from, value=30.0, coverage=100.0):
    r = {"period": {"datetimeFrom": {"local": local_from, "utc": utc_from}}, "value": value}
    if coverage is not None:
        r["coverage"] = {"percentComplete": coverage}
    return r


# 2026-10-18 00:00 in Lagos (UTC+1) is 2026-10-17 23:00 UTC
LAGOS_DAY = record("2026-10-18T00:00:00+01:00", "2026-10-17T23:00:00Z")


def test_day_is_the_local_calendar_date_not_the_utc_date():
    row = ingest.to_daily_row(1, 2, "Agege", LAGOS_DAY, NOW)
    assert row[3] == "2026-10-18"
    # The bug that was fixed: slicing the UTC timestamp is one day early
    assert LAGOS_DAY["period"]["datetimeFrom"]["utc"][:10] == "2026-10-17"


def test_row_has_the_columns_in_table_order():
    row = ingest.to_daily_row(11, 22, "Agege", record("2026-10-18T00:00:00+01:00", "x", 31.2, 87.5), NOW)
    assert row == [11, 22, "Agege", "2026-10-18", 31.2, 87.5, NOW]


def test_missing_coverage_becomes_null():
    row = ingest.to_daily_row(1, 2, "Agege", record("2026-10-18T00:00:00+01:00", "x", coverage=None), NOW)
    assert row[5] is None


@pytest.fixture
def con():
    c = duckdb.connect(":memory:")
    ingest.init(c)
    return c


def fake_api(days_by_sensor, no_pm25=()):
    """Stands in for ingest.get_json: one pm25 sensor per station (sensor id = location id * 10)."""
    def get_json(url, headers=None):
        if "/locations/" in url:
            loc_id = int(url.rsplit("/", 1)[1])
            param = "no2" if loc_id in no_pm25 else "pm25"
            return {"results": [{"sensors": [{"id": loc_id * 10, "parameter": {"name": param}}]}]}
        sensor_id = int(url.split("/sensors/")[1].split("/")[0])
        return {"results": days_by_sensor.get(sensor_id, [])}
    return get_json


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("OPENAQ_API_KEY", "test-key")


def _days_for_all(value):
    return {loc * 10: [LAGOS_DAY | {"value": value}] for loc in ingest.STATIONS}


def test_ingest_is_idempotent_and_later_values_replace_earlier_ones(con, monkeypatch):
    monkeypatch.setattr(ingest, "get_json", fake_api(_days_for_all(10.0)))
    ingest.ingest_openaq(con, "2025-08-01", "2026-10-19")
    ingest.ingest_openaq(con, "2025-08-01", "2026-10-19")
    assert con.execute("SELECT COUNT(*) FROM raw_openaq_daily").fetchone()[0] == len(ingest.STATIONS)

    # A partial day later completes: the re-ingested value must win
    monkeypatch.setattr(ingest, "get_json", fake_api(_days_for_all(42.0)))
    ingest.ingest_openaq(con, "2025-08-01", "2026-10-19")
    assert con.execute("SELECT COUNT(*) FROM raw_openaq_daily").fetchone()[0] == len(ingest.STATIONS)
    assert con.execute("SELECT DISTINCT value FROM raw_openaq_daily").fetchall() == [(42.0,)]
    assert con.execute("SELECT DISTINCT day FROM raw_openaq_daily").fetchall() == [(date(2026, 10, 18),)]


def test_station_without_a_pm25_sensor_is_skipped(con, monkeypatch, capsys):
    skipped = next(iter(ingest.STATIONS))
    monkeypatch.setattr(ingest, "get_json", fake_api(_days_for_all(10.0), no_pm25=(skipped,)))
    ingest.ingest_openaq(con, "2025-08-01", "2026-10-19")
    names = {n for (n,) in con.execute("SELECT DISTINCT location_name FROM raw_openaq_daily").fetchall()}
    assert ingest.STATIONS[skipped] not in names and len(names) == len(ingest.STATIONS) - 1
    assert "no pm25 sensor, skipped" in capsys.readouterr().out


def test_a_day_with_no_coverage_value_never_reaches_clean_daily(con, monkeypatch):
    days = {loc * 10: [record("2026-10-18T00:00:00+01:00", "x", 30.0, coverage=None)]
            for loc in ingest.STATIONS}
    monkeypatch.setattr(ingest, "get_json", fake_api(days))
    ingest.ingest_openaq(con, "2025-08-01", "2026-10-19")
    for stmt in open("sql/views.sql", encoding="utf-8").read().split(";"):
        if stmt.strip():
            con.execute(stmt)
    assert con.execute("SELECT COUNT(*) FROM clean_daily").fetchone()[0] == 0


def test_weather_rows_are_upserted(con, monkeypatch):
    payload = {"daily": {"time": ["2026-10-01", "2026-10-02"],
                         "temperature_2m_mean": [27.0, 28.0], "precipitation_sum": [0.0, 5.0],
                         "wind_speed_10m_max": [12.0, 14.0], "wind_direction_10m_dominant": [200, 210]}}
    monkeypatch.setattr(ingest, "get_json", lambda url, headers=None: payload)
    ingest.ingest_weather(con, "2026-10-01", "2026-10-02")
    ingest.ingest_weather(con, "2026-10-01", "2026-10-02")
    assert con.execute("SELECT COUNT(*) FROM raw_weather_daily").fetchone()[0] == 2
    assert con.execute("SELECT precip_sum FROM raw_weather_daily WHERE day = '2026-10-02'").fetchone()[0] == 5.0