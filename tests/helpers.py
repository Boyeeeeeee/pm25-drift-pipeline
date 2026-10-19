import numpy as np
import pandas as pd


class StubModel:
    """Stands in for the trained model: always predicts the same day-to-day change."""

    def __init__(self, delta=0.0):
        self.delta = delta

    def predict(self, X):
        return np.full(len(X), self.delta)


def make_bundle(delta=-2.0, trained_through="2026-10-14"):
    return {"model": StubModel(delta),
            "features": ["pm25_today", "pm25_yesterday", "station"],
            "station_map": {"Agege": 0, "Dolphin": 1},
            "trained_through": trained_through}


def make_clean(rows):
    """rows: (station, 'YYYY-MM-DD', pm25) tuples, shaped like data.load_clean()."""
    df = pd.DataFrame(rows, columns=["location_name", "day", "pm25"])
    df["day"] = pd.to_datetime(df["day"])
    return df


def make_pairs(stations, days, actual=20.0, pred=20.0, persist=20.0,
               trained="2026-10-14", start="2026-10-01"):
    """Logged forecasts plus matching actuals, one per station per day."""
    d0 = pd.Timestamp(start)
    p_rows, a_rows = [], []
    for s in stations:
        for i in range(days):
            day = d0 + pd.Timedelta(days=i)
            p_rows.append((s, day.date(), (day - pd.Timedelta(days=1)).date(),
                           pred, persist, trained))
            a_rows.append((s, day, actual))
    p = pd.DataFrame(p_rows, columns=["location_name", "forecast_for", "made_from",
                                      "pred", "persist", "model_trained_through"])
    a = pd.DataFrame(a_rows, columns=["location_name", "day", "pm25"])
    return p, a
