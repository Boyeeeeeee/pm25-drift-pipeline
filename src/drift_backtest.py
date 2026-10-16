import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingRegressor

from data import load_features

df = load_features()
FEATS = ["pm25_today", "pm25_yesterday", "station"]
CUTOFF = pd.Timestamp("2025-11-01")

# Reference window: labels strictly before the cutoff month
ref = df[df["day"] < CUTOFF - pd.Timedelta(days=1)]
print(f"Reference rows: {len(ref)}, median PM2.5: {ref['pm25_today'].median():.1f}")

m = HistGradientBoostingRegressor(loss="absolute_error", max_depth=3,
                                  learning_rate=0.05, max_iter=200, random_state=0)
m.fit(ref[FEATS], ref["target_next_day"] - ref["pm25_today"])

rows = []
for month, test in df[df["day"] >= CUTOFF].groupby("month"):
    pred = test["pm25_today"].to_numpy() + m.predict(test[FEATS])
    ks = ks_2samp(ref["pm25_today"], test["pm25_today"])
    rows.append({
        "month": str(month),
        "n": len(test),
        "median": test["pm25_today"].median(),
        "ks_stat": ks.statistic,
        "mae_persist": (test["target_next_day"] - test["pm25_today"]).abs().mean(),
        "mae_frozen": (test["target_next_day"] - pred).abs().mean(),
    })

out = pd.DataFrame(rows)
out["frozen_gain"] = out["mae_persist"] - out["mae_frozen"]
print(out.round(2).to_string(index=False))