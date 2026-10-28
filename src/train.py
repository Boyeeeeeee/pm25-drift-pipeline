import json
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib
from pathlib import Path

from data import load_features

FEATS = ["pm25_today", "pm25_yesterday", "station"]
PARAMS = dict(loss="absolute_error", max_depth=3, learning_rate=0.05,
              max_iter=200, random_state=0)
HOLDOUT_DAYS = 60

df = load_features().sort_values(["day", "location_name"])
last_day = df["day"].max()
split = last_day - pd.Timedelta(days=HOLDOUT_DAYS)
train = df[df["day"] < split - pd.Timedelta(days=1)]
hold = df[df["day"] >= split]


def fit(frame):
    m = HistGradientBoostingRegressor(**PARAMS)
    m.fit(frame[FEATS], frame["target_next_day"] - frame["pm25_today"])
    return m


# Honest holdout metrics: model trained on data before the last 60 days
m_hold = fit(train)
pred = hold["pm25_today"].to_numpy() + m_hold.predict(hold[FEATS])
mae_model = float((hold["target_next_day"] - pred).abs().mean())
mae_persist = float((hold["target_next_day"] - hold["pm25_today"]).abs().mean())

# Production model: refit on everything available
final = fit(df)
station_map = df.groupby("location_name")["station"].first().to_dict()
Path("models").mkdir(exist_ok=True)
joblib.dump({"model": final, "features": FEATS, "station_map": station_map,
             "trained_through": str(last_day.date())}, "models/latest.joblib")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("pm25-next-day")
with mlflow.start_run(run_name=f"train-{last_day.date()}"):
    mlflow.log_params({**PARAMS, "features": ",".join(FEATS), "target": "delta",
                       "holdout_days": HOLDOUT_DAYS, "train_rows": len(df),
                       "last_data_day": str(last_day.date())})
    mlflow.log_metrics({"holdout_mae_model": mae_model,
                        "holdout_mae_persist": mae_persist,
                        "holdout_gain": mae_persist - mae_model,
                        "holdout_rows": len(hold)})
    mlflow.log_dict(station_map, "station_map.json")
    df[["location_name", "day", "pm25_today"]].to_csv("reference.csv", index=False)
    mlflow.log_artifact("reference.csv")
    mlflow.sklearn.log_model(
        final,
        name="model",
        skops_trusted_types=["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"],
    )
    print(json.dumps({"holdout_mae_model": round(mae_model, 2),
                      "holdout_mae_persist": round(mae_persist, 2),
                      "rows": len(df)}, indent=2))