import os
from datetime import date, datetime, timedelta, timezone

import duckdb
import requests
from dotenv import load_dotenv
load_dotenv()

API = "https://api.openaq.org/v3"
STATIONS = {
    5516343: "Justrite - Ogolonto",
    5038498: "Oshodi Bus Terminal 3",
    5046018: "Agege Redline Train Station",
    4903422: "Dolphin Estate - Lagos",
}
LAT, LON = 6.5244, 3.3792
START = "2025-08-01"


def get_json(url, headers=None):
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def init(con):
    con.execute("""CREATE TABLE IF NOT EXISTS raw_openaq_daily (
        sensor_id BIGINT, location_id BIGINT, location_name VARCHAR,
        day DATE, value DOUBLE, pct_complete DOUBLE, ingested_at TIMESTAMP,
        PRIMARY KEY (sensor_id, day))""")
    con.execute("""CREATE TABLE IF NOT EXISTS raw_weather_daily (
        day DATE PRIMARY KEY, temp_mean DOUBLE, precip_sum DOUBLE,
        wind_max DOUBLE, wind_dir DOUBLE, ingested_at TIMESTAMP)""")


def ingest_openaq(con, start, end):
    h = {"X-API-Key": os.environ["OPENAQ_API_KEY"]}
    now = datetime.now(timezone.utc)
    for loc_id, name in STATIONS.items():
        loc = get_json(f"{API}/locations/{loc_id}", h)["results"][0]
        sensor = next((s for s in loc["sensors"] if s["parameter"]["name"] == "pm25"), None)
        if sensor is None:
            print(f"{name}: no pm25 sensor, skipped")
            continue
        url = f"{API}/sensors/{sensor['id']}/days?datetime_from={start}&datetime_to={end}&limit=1000"
        rows = get_json(url, h)["results"]
        for r in rows:
            con.execute(
                "INSERT OR REPLACE INTO raw_openaq_daily VALUES (?,?,?,?,?,?,?)",
                [sensor["id"], loc_id, name, r["period"]["datetimeFrom"]["local"][:10],
                 r["value"], (r.get("coverage") or {}).get("percentComplete"), now],
            )
        print(f"{name}: {len(rows)} daily rows upserted")


def ingest_weather(con, start, end):
    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={LAT}&longitude={LON}&start_date={start}&end_date={end}"
           "&daily=temperature_2m_mean,precipitation_sum,wind_speed_10m_max,wind_direction_10m_dominant"
           "&timezone=Africa%2FLagos")
    d = get_json(url)["daily"]
    now = datetime.now(timezone.utc)
    for i, day in enumerate(d["time"]):
        con.execute(
            "INSERT OR REPLACE INTO raw_weather_daily VALUES (?,?,?,?,?,?)",
            [day, d["temperature_2m_mean"][i], d["precipitation_sum"][i],
             d["wind_speed_10m_max"][i], d["wind_direction_10m_dominant"][i], now],
        )
    print(f"weather: {len(d['time'])} daily rows upserted")


if __name__ == "__main__":
    con = duckdb.connect("data/pm25.duckdb")
    init(con)
    today = date.today()
    ingest_openaq(con, START, today.isoformat())
    ingest_weather(con, START, (today - timedelta(days=5)).isoformat())
    print(con.execute("""SELECT location_name, COUNT(*), MIN(day), MAX(day)
                         FROM raw_openaq_daily GROUP BY 1 ORDER BY 1""").fetchall())