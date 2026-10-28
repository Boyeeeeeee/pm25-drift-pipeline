# Lagos PM2.5 forecasting pipeline

![tests](https://github.com/Boyeeeeeee/pm25-drift-pipeline/actions/workflows/tests.yml/badge.svg)

A scheduled pipeline that pulls Lagos air-quality and weather data, forecasts next-day PM2.5, tracks every training run, and watches for input drift and model degradation.

The point of this project is the pipeline around the model: idempotent ingestion, SQL cleaning layers, experiment tracking, scheduled runs on GitHub Actions, and monitoring rules that were written down before the data they will be judged on arrived. The model is deliberately modest, and the results below say exactly how modest.

**Live results:** see [SCOREBOARD.md](SCOREBOARD.md), rebuilt by every daily run.

## Results

All numbers come from scripts in this repo (`src/baseline.py`, `src/model.py`, `src/drift_backtest.py`, `src/train.py`). The baseline is **persistence** (tomorrow equals today). Error is mean absolute error (MAE) in µg/m³, pooled over four stations. Results were computed on a data snapshot; re-running after new ingests will shift them slightly.

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

The reference window is 299 rows (Aug to Oct 2025, median 31.1 µg/m³). The model is never retrained. `ks_stat` is the Kolmogorov-Smirnov statistic between the reference and each month's PM2.5 distribution. `gain` is persistence MAE minus frozen-model MAE (negative means the model lost). The 2026-10 row is a partial month.

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
| 2026-10 (partial) | 29.7 | 0.11 | 0.85 |

What this shows and does not show:

- The frozen model beat persistence in 11 of 12 months. Its only loss was January 2026, the harmattan peak, which was also the month with by far the largest distribution shift.
- That is **one degraded month**, not a validated monitor. September 2026 had a KS of 0.33 from a shift *downward* and the model was fine, so a low threshold would have raised false alarms.
- The most plausible reason for the January loss is that tree models cannot predict above the levels they saw in training. Predicting the day-to-day change reduced the gap but did not remove it. This is a hypothesis from one winter.

## Monitoring rules

Registered in `config/monitor.json` and committed on **2026-10-16 at 16:56 (UTC+1)**, before any 2026-27 harmattan data existed. The ground-truth rule's parameters (14-day window, minimum 28 scored forecasts) were added in a later commit at 22:58 the same evening, about five minutes before the first live forecast was logged at 23:03. The KS rule was left unchanged in a commit at 23:00. Commit timestamps are set locally; the push history on GitHub is the independent record.

1. **Input drift (leading indicator):** KS statistic between the trailing 30 days of PM2.5 and the training reference, alert at **0.5** (minimum 60 rows).
2. **Model degradation (ground truth):** alert when the live model's trailing 14-day MAE is worse than persistence. Forecasts are logged before the actuals exist, and the first forecast for a given station and day is never overwritten.

Caveats stated in advance:

- The 0.5 threshold was picked after looking at the backtest above, against a reference window that did not contain a harmattan. The deployed model's reference now includes last winter, so the same shift produces a lower KS, and rule 1 may never fire. If that happens, the finding is that the rule needs recalibrating.
- The reference is the full training data of the current production model, and `train.py` rewrites it at every retrain (weekly). So rule 1's baseline moves with each retrain, while its 0.5 threshold was chosen against one fixed historical reference.
- Rule 2 is the one to trust more, but it is coarse early on: the first 28 scored forecasts are about 7 days of four stations that share the same weather, so they are not 28 independent samples. A single weather regime could trip the rule or hide a real problem.
- Results from the 2026-27 season will be added to the live results log below, including misses and false alarms.

## Live forecast log

`src/predict_daily.py` logs a next-day PM2.5 forecast for each station from the latest ingested data, next to the persistence forecast (tomorrow = today). The first forecast for a station-day is never overwritten (`INSERT OR IGNORE` on `(location_name, forecast_for)`). A forecast is skipped if the input data is stale or its target day has already passed, so the log only contains forecasts made on or before the target day. Days dated today are never used as input, because a day in progress is only a partial average.

`src/evaluate_live.py` compares model MAE with persistence MAE over the last 14 days. It evaluates once 28 scored forecasts exist; the first evaluation was on 2026-10-26.

Each forecast stores `model_trained_through`, the last data day the model saw. Because `train.py` overwrites `models/latest.joblib` at each retrain, forecasts from different model versions end up in the same log, and live MAE is pooled across them unless filtered by that column. `live_eval` also records `n_models`, the number of distinct model versions inside each evaluation window, so a window that spans a retrain is visible.

`SCOREBOARD.md` lists every forecast that has an actual, including misses, so the live record can be checked without running anything.

**Where the log is written.** Forecasts for 2026-10-16, 10-19 and 10-20 were produced on my laptop. From 2026-10-21 the `daily-forecast` GitHub Actions workflow produces them (its first scheduled run was on 2026-10-20). It is scheduled for 07:17 and 11:17 Lagos time and keeps the log in `logs/predictions.csv`, `logs/drift_log.csv` and `logs/live_eval.csv`, committing them after each run, so each commit time comes from GitHub. The workflow fails visibly if no forecast exists for the day. In practice GitHub starts scheduled runs hours late (on 2026-10-21 the 07:17 slot ran at about 12:49 and the 11:17 slot at about 16:14), so forecasts are made around midday or later on the target day, from data through the day before. Part of the target day has therefore already passed when a forecast is made.

**Timestamps.** `made_at` is UTC for every forecast made by GitHub Actions (from 2026-10-21). The earlier forecasts, made on my laptop (10-16, 10-19 and 10-20), stored Lagos local time (UTC+1) by mistake; they are left unchanged so the history is not edited.

**Model versions.** The model is retrained every Monday at a nominal 12:47 Lagos time by the `weekly-retrain` workflow, which commits the new `models/latest.joblib` and `reference.csv` with the holdout MAE in the commit message. Forecasts up to 2026-10-20 come from the model trained through 2026-10-14. The first retrain was run by hand on 2026-10-20 (trained through 2026-10-18, used from 2026-10-21), and the first scheduled retrain ran on Monday 2026-10-26.

Log history and gaps, stated as they happened:

- The log started on **2026-10-16**. The three forecasts for that day were made at about 23:00 on the target day, from data through 10-15, so they are late rather than made in advance. They are kept and flagged.
- One retrospective forecast (Dolphin Estate, target 10-15, made after that day ended) was deleted.
- There are **no forecasts for 2026-10-17 and 2026-10-18**: the pipeline only ran while my laptop was on. These cannot be backfilled honestly.
- The first forecasts made on the target day before it was complete are for **2026-10-19**.
- No gaps since 2026-10-19 as of the entry below.

## Live results log

Dated entries, newest last. Misses are reported alongside the wins; the full table is in [SCOREBOARD.md](SCOREBOARD.md).

**Entry of 2026-10-28 (data through 2026-10-27).** Rule 2 was first evaluated on 2026-10-26 (31 scored forecasts, as of 10-25: model MAE 7.11, persistence 8.49) and again on 2026-10-27 (35 scored, as of 10-26, nine forecast days: model 6.66, persistence 8.08). No alert either time. The model was closer on 20 of 35 forecasts and on 6 of 9 days, and lost on 10-19, 10-23 and 10-24. One day, 2026-10-22, accounts for about 72% of the model's total advantage (persistence missed by 14.83 and the model by 5.83, the day after a sharp rise at all four stations on 10-21); without it the figures are 6.77 and 7.21. Rule 1 (input drift) stayed between KS 0.19 and 0.23 for 10-21 to 10-26, well below its 0.5 threshold. The weekly retrain of 2026-10-26 had a 60-day holdout MAE of 5.67 against 6.26 for persistence on 1,556 rows.

This is not validation: nine forecast days, four stations that share the same weather, and no harmattan yet.

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
                                  v
                          scoreboard.py ──> SCOREBOARD.md

logstore.py: logs/*.csv <-> DuckDB (the forecast log is kept in git)
FastAPI (serve.py): /predict, /forecasts/latest, /monitor/status, /health

GitHub Actions: daily-forecast (ingest, forecast, evaluate, drift check, scoreboard,
                commit logs), weekly-retrain, tests (pytest and a Docker smoke test)
Prefect flows (src/flows.py): a local alternative, not used for the live log
```

## Data

- **PM2.5:** [OpenAQ](https://openaq.org) v3 API, daily aggregates for four Lagos stations (Justrite - Ogolonto, Oshodi Bus Terminal 3, Agege Redline Train Station, Dolphin Estate). Days with under 75% hourly coverage are dropped (99 days at the time of the evaluation snapshot).
- **Weather:** [Open-Meteo](https://open-meteo.com) historical archive, daily values for Lagos. Used in experiments only.
- At the evaluation snapshot: 1,682 station-days ingested, 1,517 usable labelled rows after cleaning and the next-day join. The database keeps growing as ingest runs.
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
python src/predict_daily.py   # forecast for today (the live log is written by GitHub Actions; a local run makes a separate log)
python src/evaluate_live.py   # score logged forecasts against actuals
python src/monitor.py         # input-drift check
python src/scoreboard.py      # rebuild SCOREBOARD.md from the local database

mlflow ui --backend-store-uri sqlite:///mlflow.db
uvicorn serve:app --app-dir src --port 8000    # API docs at /docs
python src/flows.py           # optional local Prefect schedule (not used for the live log)
```

The OpenAQ API key is free from explore.openaq.org and is read from the `OPENAQ_API_KEY` environment variable. Run all scripts from the repo root, because paths are relative.

`monitor.py` reads `reference.csv`, which `train.py` writes. MLflow runs (`mlflow.db`, `mlruns/`) are local and not committed. `models/latest.joblib` and `reference.csv` are committed by the weekly retrain workflow so that GitHub runners can forecast.

On Windows PowerShell, set `$env:PYTHONUTF8 = "1"` if a script crashes while printing non-ASCII characters (cp1252 console encoding).

## Docker

The image serves the API only; the pipeline itself runs on GitHub Actions.

```powershell
docker build -t pm25-api .
docker run --rm -p 8000:8000 pm25-api    # docs at http://127.0.0.1:8000/docs
```

The image contains the code, the trained model (`models/latest.joblib`) and a snapshot of the committed logs (`logs/`), which are loaded into a fresh DuckDB when the container starts. It has no raw air-quality data, so `/monitor/status` reports `unavailable` for the scored-forecast count instead of a number. Rebuild the image to pick up newer forecasts and models. The container runs as a non-root user and has a health check. Dependencies are in `requirements-docker.txt`, which has to be kept in step with `requirements.txt`. CI builds the image and calls every endpoint on each push.

## Repo layout

```
src/
  ingest.py           OpenAQ + Open-Meteo to DuckDB, upserts (idempotent), retries transient errors
  data.py             shared loaders over the SQL views
  baseline.py         persistence baseline
  model.py            walk-forward evaluation
  drift_backtest.py   frozen-model drift backtest
  train.py            training + MLflow logging
  predict_daily.py    forecast log
  evaluate_live.py    live MAE vs persistence
  monitor.py          input-drift check
  logstore.py         logs/*.csv <-> DuckDB
  scoreboard.py       writes SCOREBOARD.md
  flows.py            Prefect flows (local alternative)
  serve.py            FastAPI service
sql/views.sql         clean_daily and features views
config/monitor.json   registered monitoring rules
logs/                 forecast log, drift log, live evaluation (committed by the daily workflow)
models/latest.joblib  current model (committed by the weekly retrain)
reference.csv         training reference for the drift rule (committed by the weekly retrain)
SCOREBOARD.md         every scored live forecast
tests/                test suite
.github/workflows/    tests.yml, daily-forecast.yml, weekly-retrain.yml
Dockerfile
requirements.txt, requirements-dev.txt, requirements-docker.txt
```

## Tests

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
```

The suite (78 tests) uses synthetic data and a stub model, so it needs no API key, network access or real database. It covers:

- the forecast guards: unknown station, missing previous day, stale data, target already past, and a partial current day being ignored
- the first forecast being kept, and `made_at` being stored in UTC
- the 28-forecast and strict-worse-than-persistence rules in `evaluate_live.py`
- the inclusive `KS >= 0.5` rule and trailing-window logic in `monitor.py`
- the 75% coverage filter and the gap-safe feature join in SQL
- ingestion: days bucketed by Lagos local date, idempotent upserts where later values replace earlier ones, and retries of transient errors
- the log CSV round trip, and the scoreboard
- the API's status codes, including a deployment that has only the logs

One test asserts the values in `config/monitor.json` match the rules registered on 2026-10-16, so changing a registered rule has to be done on purpose, and the README has to change with it.

## Status

- [x] Idempotent ingestion, SQL cleaning views
- [x] Persistence baseline, walk-forward evaluation, drift backtest
- [x] MLflow-tracked training
- [x] Input-drift monitor with pre-registered rules
- [x] Prefect flows (failure and retry behaviour checked; now a local alternative)
- [x] Forecast log (idempotent on re-run; written by GitHub Actions since 2026-10-21)
- [x] Live evaluation (rule 2 first evaluated on 2026-10-26, no alert)
- [x] FastAPI service (all four endpoints checked; `/monitor/status` shows the last logged results, not a live computation)
- [x] Tests (78)
- [x] CI (tests, plus a Docker build and endpoint smoke test, on every push)
- [x] Weekly retrain (first scheduled run on 2026-10-26)
- [x] Dockerfile (API image)
- [x] Public scoreboard of every scored live forecast (SCOREBOARD.md, regenerated daily)
- [x] Always-on schedule (GitHub Actions `daily-forecast`; first scheduled run 2026-10-20)
- [ ] Live validation through the 2026-27 harmattan, results reported here either way

## Known limitations

- Single-point daily features only. No forecast weather for the target day, no spatial features.
- Pooled KS across stations is a coarse signal.
- The live log has a gap (10-17 and 10-18) and three late forecasts on its first day, described above.
- GitHub starts scheduled runs hours late, so forecasts are made around midday or later on the target day. A day is lost only if neither of its two runs succeeds.
- The GitHub Actions workflow is the only writer of the live log. Running `predict_daily.py` locally creates a separate local log that is not committed.
- Retraining is automatic and unreviewed; rule 2 (live MAE against persistence) is the safeguard, not a human check.
- DuckDB allows one writer process at a time, so the API opens short-lived connections and the pipeline should not run while another process holds the file open. A server database would be the next step for real deployment.
- The model file is stored with joblib, which is pickle-based. It is only ever loaded from this project's own training output.
- The API loads the model once at startup, so it must be restarted after each retrain; `/health` reports which training date it is serving.
- The Docker image is a snapshot of the logs and model at build time, and its dependency pins are a second list to keep in step with `requirements.txt`.