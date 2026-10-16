import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

con = duckdb.connect("data/pm25.duckdb")
for stmt in open("sql/views.sql", encoding="utf-8").read().split(";"):
    if stmt.strip():
        con.execute(stmt)
df = con.sql("SELECT * FROM features").df()
df["day"] = pd.to_datetime(df["day"])
df["month"] = df["day"].dt.to_period("M")
df["wd_sin"] = np.sin(np.deg2rad(df["wind_dir"]))
df["wd_cos"] = np.cos(np.deg2rad(df["wind_dir"]))
df["station"] = df["location_name"].astype("category").cat.codes

AR = ["pm25_today", "pm25_yesterday", "station"]
WX = ["temp_mean", "precip_sum", "wind_max", "wd_sin", "wd_cos"]

def fit_predict(train, test, feats):
    m = HistGradientBoostingRegressor(loss="absolute_error", max_depth=3,
                                      learning_rate=0.05, max_iter=200, random_state=0)
    m.fit(train[feats], train["target_next_day"] - train["pm25_today"])
    return test["pm25_today"].to_numpy() + m.predict(test[feats])

parts = []
for month in sorted(df["month"].unique()):
    start = month.start_time
    # Train only on rows whose label day is strictly before the test month,
    # so no training label equals a test feature value.
    train = df[df["day"] < start - pd.Timedelta(days=1)]
    test = df[df["month"] == month]
    if len(train) < 200 or len(test) == 0:
        continue
    t = test.copy()
    t["persist"] = t["pm25_today"]
    t["gbm_ar"] = fit_predict(train, test, AR)
    t["gbm_wx"] = fit_predict(train, test, AR + WX)
    parts.append(t)

res = pd.concat(parts)
for c in ["persist", "gbm_ar", "gbm_wx"]:
    res[f"ae_{c}"] = (res["target_next_day"] - res[c]).abs()

by_month = res.groupby("month")[["ae_persist", "ae_gbm_ar", "ae_gbm_wx"]].mean().round(2)
by_month.insert(0, "n", res.groupby("month").size())
print("MAE by test month (walk-forward):")
print(by_month.to_string())
print("\nOverall MAE:")
print(res[["ae_persist", "ae_gbm_ar", "ae_gbm_wx"]].mean().round(2).to_string())

rng = np.random.default_rng(0)
for c in ["gbm_ar", "gbm_wx"]:
    d = (res["ae_persist"] - res[f"ae_{c}"]).groupby(res["day"]).mean().to_numpy()
    boot = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    lo, hi = np.percentile(boot, [2.5, 97.5]).round(2)
    print(f"{c}: MAE improvement vs persistence, 95% CI [{lo}, {hi}]")