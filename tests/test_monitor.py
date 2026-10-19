import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from helpers import make_clean
from monitor import compute_drift, log_drift

RULES = dict(window_days=30, min_rows=60, ks_threshold=0.5)


def _ref(seed=0, loc=30.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"day": pd.date_range("2025-08-01", periods=300),
                         "pm25_today": rng.normal(loc, 8, 300)})


def _cur(values_by_day, stations=("A", "B", "C")):
    rows = [(s, d, v) for s in stations for d, v in values_by_day]
    return make_clean(rows)


def _days(n, loc, seed, start="2026-09-01"):
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, periods=n)
    return list(zip(days.strftime("%Y-%m-%d"), rng.normal(loc, 8, n)))


def test_too_few_rows_in_window_is_not_computed():
    cur = _cur(_days(10, 30, 1), stations=("A", "B"))  # 20 rows
    r = compute_drift(_ref(), cur, **RULES)
    assert r["status"] == "insufficient" and r["n"] == 20


def test_no_alert_when_the_distribution_is_unchanged():
    r = compute_drift(_ref(), _cur(_days(30, 30, 2)), **RULES)
    assert r["status"] == "computed"
    assert r["ks"] < 0.5 and r["alert"] is False


def test_alert_when_the_distribution_shifts_up():
    r = compute_drift(_ref(), _cur(_days(30, 70, 3)), **RULES)
    assert r["ks"] > 0.9 and r["alert"] is True


def test_threshold_is_inclusive():
    r = compute_drift(_ref(), _cur(_days(30, 45, 4)), **RULES)
    at = compute_drift(_ref(), _cur(_days(30, 45, 4)), RULES["window_days"],
                       RULES["min_rows"], ks_threshold=r["ks"])
    above = compute_drift(_ref(), _cur(_days(30, 45, 4)), RULES["window_days"],
                          RULES["min_rows"], ks_threshold=r["ks"] + 1e-6)
    assert at["alert"] is True and above["alert"] is False


def test_only_the_trailing_window_is_compared():
    old = _days(30, 200, 5, start="2026-08-01")   # wildly different, but outside the window
    new = _days(30, 30, 6, start="2026-09-01")
    r = compute_drift(_ref(), _cur(old + new), **RULES)
    assert r["n"] == 90 and r["alert"] is False


def test_drift_row_is_replaced_for_the_same_day():
    r = compute_drift(_ref(), _cur(_days(30, 30, 7)), **RULES)
    con = duckdb.connect(":memory:")
    log_drift(con, r)
    log_drift(con, r)
    assert con.execute("SELECT COUNT(*) FROM drift_log").fetchone()[0] == 1


def test_registered_monitoring_rules_are_unchanged():
    # The rules were registered on 2026-10-16, before the 2026-27 harmattan, and the
    # README says so. Changing a value here is changing the registered rule: do it in
    # a separate, clearly described commit and update the README in the same change.
    cfg = json.loads(Path("config/monitor.json").read_text(encoding="utf-8"))
    assert cfg["ks_threshold"] == 0.5
    assert cfg["window_days"] == 30
    assert cfg["min_rows"] == 60
    assert cfg["gt_window_days"] == 14
    assert cfg["gt_min_rows"] == 28
