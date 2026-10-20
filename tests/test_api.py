import duckdb
import pytest
from fastapi.testclient import TestClient

import serve
from helpers import make_bundle


@pytest.fixture
def client(tmp_path, monkeypatch):
    path = str(tmp_path / "api.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE raw_openaq_daily (location_name VARCHAR, day DATE, value DOUBLE, pct_complete DOUBLE)")
    con.execute("CREATE TABLE raw_weather_daily (day DATE, temp_mean DOUBLE, precip_sum DOUBLE, wind_max DOUBLE, wind_dir DOUBLE)")
    for stmt in open("sql/views.sql", encoding="utf-8").read().split(";"):
        if stmt.strip():
            con.execute(stmt)
    # Actuals exist for the three 10-16 forecasts only
    con.executemany("INSERT INTO raw_openaq_daily VALUES (?,?,?,?)",
                    [("Agege", "2026-10-16", 33.0, 100.0), ("Dolphin", "2026-10-16", 30.0, 100.0),
                     ("Justrite", "2026-10-16", 25.0, 100.0)])
    con.execute("""CREATE TABLE predictions (
        location_name VARCHAR, forecast_for DATE, made_from DATE, pred DOUBLE, persist DOUBLE,
        model_trained_through DATE, made_at TIMESTAMP, PRIMARY KEY (location_name, forecast_for))""")
    con.executemany("INSERT INTO predictions VALUES (?,?,?,?,?,?,now())",
                    [("Agege", "2026-10-16", "2026-10-15", 31.94, 33.5, "2026-10-14"),
                     ("Dolphin", "2026-10-16", "2026-10-15", 28.7, 36.1, "2026-10-14"),
                     ("Justrite", "2026-10-16", "2026-10-15", 26.3, 23.7, "2026-10-14"),
                     ("Dolphin", "2026-10-19", "2026-10-18", 15.14, 18.6, "2026-10-14"),
                     ("Agege", "2026-10-19", "2026-10-18", 30.7, 31.2, "2026-10-14")])
    con.close()

    monkeypatch.setattr(serve, "DB", path)
    monkeypatch.setattr(serve.joblib, "load", lambda _: make_bundle(delta=-2.0))
    with TestClient(serve.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_trained_through": "2026-10-14",
                        "stations": ["Agege", "Dolphin"]}


def test_predict_returns_today_plus_model_change(client):
    r = client.post("/predict", json={"station": "Agege", "pm25_today": 31.2, "pm25_yesterday": 35.0})
    assert r.status_code == 200
    body = r.json()
    assert body["pm25_next_day"] == 29.2
    assert body["persistence_baseline"] == 31.2


def test_predict_unknown_station_is_404_not_500(client):
    r = client.post("/predict", json={"station": "Nowhere", "pm25_today": 30, "pm25_yesterday": 30})
    assert r.status_code == 404
    assert "Agege" in r.json()["detail"]


@pytest.mark.parametrize("body", [
    {"station": "Agege", "pm25_today": -5, "pm25_yesterday": 30},
    {"station": "Agege", "pm25_today": 30},
    {"station": "Agege", "pm25_today": "abc", "pm25_yesterday": 30},
    {"station": "Agege", "pm25_today": 1001, "pm25_yesterday": 30},
])
def test_predict_rejects_bad_input_with_422(client, body):
    assert client.post("/predict", json=body).status_code == 422


def test_latest_forecasts_are_only_the_newest_day_with_plain_dates(client):
    rows = client.get("/forecasts/latest").json()
    assert [r["location_name"] for r in rows] == ["Agege", "Dolphin"]
    assert {r["forecast_for"] for r in rows} == {"2026-10-19"}
    assert rows[1]["pred"] == 15.1


def test_monitor_status_says_when_there_is_not_enough_data(client):
    r = client.get("/monitor/status").json()
    assert r["input_drift"] == []
    assert r["live_eval"]["status"] == "insufficient_data"
    assert r["live_eval"]["scored_forecasts"] == 3
    assert r["live_eval"]["required"] == 28


def test_monitor_status_reports_a_logged_evaluation(client, tmp_path):
    con = duckdb.connect(serve.DB)
    con.execute("""CREATE TABLE live_eval (as_of DATE PRIMARY KEY, n INTEGER, mae_model DOUBLE,
                   mae_persist DOUBLE, alert BOOLEAN, n_models INTEGER, logged_at TIMESTAMP)""")
    con.execute("INSERT INTO live_eval VALUES ('2026-10-26', 28, 5.0, 6.0, false, 1, now())")
    con.close()
    r = client.get("/monitor/status").json()
    assert r["live_eval"]["status"] == "evaluated"
    assert r["live_eval"]["history"][0]["as_of"] == "2026-10-26"


@pytest.fixture
def logs_only_client(tmp_path, monkeypatch):
    """What the Docker image has: the committed logs imported into a fresh database, no raw data."""
    from logstore import import_logs, export_logs

    seed = duckdb.connect(":memory:")
    from predict_daily import log_forecasts
    from datetime import date
    log_forecasts(seed, [{"location_name": "Agege", "forecast_for": date(2026, 10, 20),
                          "made_from": date(2026, 10, 19), "pred": 38.0, "persist": 38.8}], "2026-10-18")
    export_logs(seed, tmp_path / "logs")

    path = str(tmp_path / "fresh.duckdb")
    con = duckdb.connect(path)
    import_logs(con, tmp_path / "logs")
    con.close()
    monkeypatch.setattr(serve, "DB", path)
    monkeypatch.setattr(serve.joblib, "load", lambda _: make_bundle(delta=-2.0))
    with TestClient(serve.app) as c:
        yield c


def test_logs_only_deployment_serves_forecasts(logs_only_client):
    rows = logs_only_client.get("/forecasts/latest").json()
    assert rows == [{"location_name": "Agege", "forecast_for": "2026-10-20",
                     "made_from": "2026-10-19", "pred": 38.0, "persistence": 38.8}]


def test_logs_only_deployment_does_not_claim_zero_scored_forecasts(logs_only_client):
    live = logs_only_client.get("/monitor/status").json()["live_eval"]
    assert live["status"] == "unavailable" and live["scored_forecasts"] is None


def test_missing_database_file_gives_empty_results_not_a_500(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "DB", str(tmp_path / "nope.duckdb"))
    monkeypatch.setattr(serve.joblib, "load", lambda _: make_bundle())
    with TestClient(serve.app) as c:
        assert c.get("/forecasts/latest").json() == []
        assert c.get("/health").status_code == 200