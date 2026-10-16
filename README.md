# Lagos PM2.5 forecasting pipeline

A scheduled pipeline that pulls Lagos air-quality and weather data, forecasts next-day PM2.5, tracks every training run, and watches for input drift and model degradation.

The point of this project is the pipeline around the model: idempotent ingestion, SQL cleaning layers, experiment tracking, orchestration with retries, and monitoring rules that were written down before the data they will be judged on arrived. The model is deliberately modest, and the results below say exactly how modest.

## Results

All numbers come from scripts in this repo (`src/baseline.py`, `src/model.py`, `src/drift_backtest.py`, `src/train.py`). The baseline is **persistence** (tomorrow equals today). Error is mean absolute error (MAE) in µg/m³, pooled over four stations.

### Walk-forward evaluation (12 test months, Nov 2025 to Oct 2026, 1,214 station-days)

For each test month the model is refit on all earlier data only.

| Model | Overall MAE |
|---|---|
| Persistence | 6.75 |
| Gradient boosting, past PM2.5 only, predicts the day-to-day change | **6.24** |
| Same, plus same-day weather features | 6.37 |

- The best model improves on persistence by about 0.5 µg/m³ (roughly 7%). A bootstrap over days gives a 95% interval of [0.31, 0.74], but it treats days as independent, so the true interval is wider.
- **Weather features did not help** for next-day prediction at a single point, so the shipped model does not use them.
- Two target formulations were tried (predict the level, then predict the change), and the second was chosen after seeing the first on the same data. The gain is probably slightly optimistic.
- A 60-day holdout in `src/train.py` gives MAE 5.92 against 6.54 for persistence. This is one window and is only a health check.

### Drift backtest (model frozen, trained on data before Nov 2025)

The reference window is 299 rows (Aug to Oct 2025, median 31.1 µg/m³). The model is never retrained. `ks_stat` is the Kolmogorov-Smirnov statistic between the reference and each month's PM2.5 distribution. `gain` is persistence MAE minus frozen-model MAE (negative means the model lost).

| Month | Median | ks_stat | gain |
|---|---|---|---|
| 2025-11 | 39.0 | 0.29 | 0.27 |
| 2025-12 | 41.2 | 0.39 | 0.14 |
| 2026-01 | 54.6 | **0.67** | **-0.66** |
| 2026-02 | 40.9 | 0.36 | 0.35 |
| 2026-03 | 32.9 | 0.15 | 0.11 |
| 2026-04 | 28.2 | 0.15 | 0.96 |
| 2026-05 | 30.7 | 0.09 | 0.79 |
| 2026-06 | 35.3 | 0.18 | 0.79 |
| 2026-07 | 33.1 | 0.12 | 0.73 |
| 2026-08 | 30.9 | 0.08 | 0.34 |
| 2026-09 | 25.1 | 0.33 | 0.09 |
| 2026-10 | 29.7 | 0.11 | 0.85 |

What this shows and does not show:

- The frozen model beat persistence in 11 of 12 months. Its only loss was January 2026, the harmattan peak, which was also the month with by far the largest distribution shift.
- That is **one degraded month**, not a validated monitor. September 2026 had a KS of 0.33 from a shift *downward* and the model was fine, so a low threshold would have raised false alarms.
- The most plausible reason for the January loss is that tree models cannot predict above the levels they saw in training. Predicting the day-to-day change reduced the gap but did not remove it. This is a hypothesis from one winter.

## Monitoring rules

Registered in `config/monitor.json` and committed on **2026-10-16**, before the 2026-27 harmattan data existed.

1. **Input drift (leading indicator):** KS statistic between the trailing 30 days of PM2.5 and the training reference, alert at **0.5** (minimum 60 rows).
2. **Model degradation (ground truth):** alert when the live model's trailing 14-day MAE is worse than persistence. Forecasts are logged before the actuals exist, and the first forecast for a given station and day is never overwritten.

Caveats stated in advance:

- The 0.5 threshold was picked after looking at the backtest above, against a reference window that did not contain a harmattan. The deployed model's reference now includes last winter, so the same shift produces a lower KS, and rule 1 may never fire. If that happens, the finding is that the rule needs recalibrating.
- Rule 2 is the one to trust. Rule 1 is an early-warning signal.
- Results from the 2026-27 season will be added here, including misses and false alarms.

## Architecture

```
OpenAQ API ──┐                                    ┌──> MLflow (params, metrics, artifacts)
             ├─> ingest.py ─> DuckDB (raw tables) ┤
Open-Meteo ──┘                    │               └──> models/latest.joblib
                                  v
                       SQL views (clean_daily, features)
                                  │
              ┌───────────────────┼────────────────────┐
              v                   v                    v
          train.py         predict_daily.py        monitor.py
                           (logs forecasts)    (input drift, KS)
                                  │
                                  v
                          evaluate_live.py ──> live_eval table
                                  │
                       FastAPI (serve.py): /predict, /forecasts/latest,
                                  /monitor/status, /health

Prefect: daily flow (ingest, forecast, evaluate, monitor), weekly flow (retrain)
```

## Data

- **PM2.5:** [OpenAQ](https://openaq.org) v3 API, daily aggregates for four Lagos stations (Justrite - Ogolonto, Oshodi Bus Terminal 3, Agege Redline Train Station, Dolphin Estate). Days with under 75% hourly coverage are dropped (99 days).
- **Weather:** [Open-Meteo](https://open-meteo.com) historical archive, daily values for Lagos. Used in experiments only.
- 1,682 station-days ingested, 1,517 usable labelled rows after cleaning and the next-day join.
- Days are bucketed by Lagos local time (UTC+1). Using UTC dates shifted every day by one.
- The OpenAQ days endpoint ignored the `datetime_from` filter, so each station's full history is fetched and start dates differ by station.

Data caveats:

- These are likely low-cost sensor networks, not reference monitors. Instrument metadata was checked for one Lagos station (AirVisual, provider IQAir), but **not individually for the four stations used here**.
- Only one harmattan season (2025-26) has been observed, so seasonal behaviour is learned from a single cycle. Day-of-year is deliberately not a feature for that reason.
- The four stations share the same days and weather, so test rows are correlated.
- Check each provider's licence before reuse. The station inspected is CC BY 4.0, and Open-Meteo asks for attribution.

## Quickstart (Windows, PowerShell)

```powershell
git clone <repo-url>
cd pm25-drift-pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env        # then put your OpenAQ API key in .env

python src/ingest.py          # backfill, safe to re-run
python src/baseline.py        # persistence baseline
python src/model.py           # walk-forward evaluation
python src/drift_backtest.py  # frozen-model drift backtest
python src/train.py           # train and log to MLflow
python src/predict_daily.py   # log tomorrow's forecasts
python src/evaluate_live.py   # score logged forecasts against actuals
python src/monitor.py         # input-drift check

mlflow ui --backend-store-uri sqlite:///mlflow.db
uvicorn serve:app --app-dir src --port 8000    # API docs at /docs
python src/flows.py           # Prefect schedule: daily 07:00, weekly Mon 08:00
```

The OpenAQ API key is free from explore.openaq.org and is read from the `OPENAQ_API_KEY` environment variable.

## Repo layout

```
src/
  ingest.py           OpenAQ + Open-Meteo to DuckDB, upserts (idempotent)
  data.py             shared loaders over the SQL views
  baseline.py         persistence baseline
  model.py            walk-forward evaluation
  drift_backtest.py   frozen-model drift backtest
  train.py            training + MLflow logging
  predict_daily.py    forecast log
  evaluate_live.py    live MAE vs persistence
  monitor.py          input-drift check
  flows.py            Prefect flows
  serve.py            FastAPI service
sql/views.sql         clean_daily and features views
config/monitor.json   registered monitoring rules
```

## Status

- [x] Idempotent ingestion, SQL cleaning views
- [x] Persistence baseline, walk-forward evaluation, drift backtest
- [x] MLflow-tracked training
- [x] Input-drift monitor with pre-registered rules
- [x] Prefect flows (failure and retry behaviour checked)
- [ ] Forecast log, live evaluation and FastAPI service (code written, being verified)
- [ ] Tests and CI
- [ ] Dockerfile
- [ ] Always-on schedule (the Prefect schedule currently only runs while my machine is on)
- [ ] Live validation through the 2026-27 harmattan, results reported here either way

## Known limitations

- Single-point daily features only. No forecast weather for the target day, no spatial features.
- Pooled KS across stations is a coarse signal.
- DuckDB allows one writer process at a time, so the API opens short-lived connections and the pipeline should not run while another process holds the file open. A server database would be the next step for real deployment.
- The model file is stored with joblib, which is pickle-based. It is only ever loaded from this project's own training output.