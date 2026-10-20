
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb

from evaluate_live import CREATE_LIVE_EVAL
from ingest import STATIONS
from monitor import CREATE_DRIFT_LOG
from predict_daily import CREATE_PREDICTIONS

DB = "data/pm25.duckdb"
LOG_DIR = "logs"

# table -> (create statement, deterministic row order so git diffs stay small)
TABLES = {
    "predictions": (CREATE_PREDICTIONS, "forecast_for, location_name"),
    "drift_log": (CREATE_DRIFT_LOG, "as_of"),
    "live_eval": (CREATE_LIVE_EVAL, "as_of"),
}


def _lit(path):
    return str(path).replace("'", "''")


def export_logs(con, out_dir=LOG_DIR):
    """Write each log table that exists to out_dir/<table>.csv. Returns {table: rows}."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    counts = {}
    for table, (_, order) in TABLES.items():
        try:
            con.execute(f"SELECT 1 FROM {table} LIMIT 1")
        except duckdb.CatalogException:
            continue
        path = Path(out_dir) / f"{table}.csv"
        con.execute(f"COPY (SELECT * FROM {table} ORDER BY {order}) "
                    f"TO '{_lit(path)}' (HEADER, DELIMITER ',')")
        counts[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return counts


def import_logs(con, in_dir=LOG_DIR):
    """Create the log tables and load any CSVs. Existing rows are never overwritten."""
    counts = {}
    for table, (create, _) in TABLES.items():
        con.execute(create)
        path = Path(in_dir) / f"{table}.csv"
        if path.exists():
            con.execute(f"INSERT OR IGNORE INTO {table} BY NAME "
                        f"SELECT * FROM read_csv('{_lit(path)}', header=true, all_varchar=true)")
        counts[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return counts


def forecasts_for(con, day):
    return con.execute("SELECT COUNT(*) FROM predictions WHERE forecast_for = ?", [day]).fetchone()[0]


def main(argv):
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd not in {"import", "export", "check"}:
        print(__doc__)
        return 2
    con = duckdb.connect(DB)
    try:
        if cmd == "import":
            print("imported:", import_logs(con))
        elif cmd == "export":
            print("exported:", export_logs(con))
        else:
            today = datetime.now(ZoneInfo("Africa/Lagos")).date()
            n = forecasts_for(con, today)
            if n == 0:
                print(f"::error::No forecast for {today} is logged; the day will be missed "
                      "unless this workflow is re-run today.")
                return 1
            if n < len(STATIONS):
                print(f"::warning::Only {n} of {len(STATIONS)} stations have a forecast for {today}.")
            print(f"{n} forecasts logged for {today}")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))