# Time Series Intelligence with Merlion

A reproducible project demonstrating **forecasting**, **anomaly detection**, **multi-model benchmarking**, and **production-like rolling backtesting** with Salesforce's [Merlion](https://github.com/salesforce/Merlion) time-series library.

The repository contains three workflows:

1. a small synthetic end-to-end demo,
2. a multi-model benchmark using real US macroeconomic data packaged with `statsmodels`, and
3. a production-like evaluator workflow with rolling retraining and sliding/expanding training windows.

## What this repository demonstrates

- Merlion `TimeSeries` conversion from pandas data
- Forecasting with multiple Merlion models
- Anomaly detection with multiple Merlion detectors
- Forecast evaluation with sMAPE and RMSE
- Anomaly evaluation with precision, recall, and F1
- Merlion `ForecastEvaluator` for historical deployment simulation
- Merlion `TSADEvaluator` for live-style anomaly detection simulation
- Train-once, expanding-window, and sliding-window retraining policies
- Model runtime measurement
- Reproducible synthetic data generation
- A real-data benchmark with no external data download
- CSV benchmark tables and PNG comparison plots

## Models benchmarked

### Forecasting

- `DefaultForecaster`
- `ARIMA`
- `ETS`
- `Prophet`

The forecasting benchmark uses quarterly US real GDP from `statsmodels.datasets.macrodata`.

### Anomaly detection

- `DefaultDetector`
- `IsolationForest`
- `StatThreshold`

Because the macroeconomic dataset does not contain anomaly labels, anomaly detection is evaluated on **real GDP growth with controlled point anomalies injected only into the held-out test period**. This keeps the underlying signal real while providing known ground truth for precision, recall, and F1.

## Project structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
└── src
    ├── __init__.py
    ├── data.py
    ├── run_demo.py
    ├── benchmark.py
    └── production_backtest.py
```

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This project pins Merlion to `salesforce-merlion==2.0.2`, the latest upstream release published by Salesforce.

## Run the synthetic demo

```bash
python -m src.run_demo
```

It creates:

- `artifacts/forecast.png`
- `artifacts/anomaly_detection.png`
- `artifacts/metrics.csv`

## Run the real-data benchmark

```bash
python -m src.benchmark
```

It creates:

- `artifacts/forecast_benchmark.csv` — forecasting model ranking by sMAPE
- `artifacts/anomaly_benchmark.csv` — detector ranking by F1
- `artifacts/forecast_benchmark.png` — actual GDP and model forecasts
- `artifacts/anomaly_benchmark.png` — true and detected anomalies

The benchmark is fault-tolerant: if an optional model fails because of a platform-specific dependency, the remaining models still run and the failure is recorded in the `status` column.

## Run the production-like backtest

```bash
python -m src.production_backtest
```

This workflow uses Merlion's evaluator layer rather than fitting a model once and forecasting the entire test set in one batch.

Three deployment policies are compared:

| Policy | Retraining | Training history |
|---|---|---|
| `train_once` | never | initial training set only |
| `expanding_annual` | approximately yearly | all history available up to each retraining point |
| `sliding_10y_annual` | approximately yearly | most recent 10 years only |

For forecasting, the workflow uses `ForecastEvaluator` with ARIMA and a roughly one-quarter prediction horizon. The evaluator walks forward through the historical test period and retrains according to the selected policy.

For anomaly detection, the workflow uses `TSADEvaluator` with Isolation Forest. Predictions are generated incrementally through the test period while the detector is periodically retrained according to the same deployment policies.

It creates:

- `artifacts/production_forecast_backtest.csv`
- `artifacts/production_anomaly_backtest.csv`
- `artifacts/production_forecast_backtest.png`
- `artifacts/production_anomaly_backtest.png`

The forecast table reports sMAPE, RMSE, runtime, retraining frequency, training-window policy, and execution status. The anomaly table reports precision, recall, F1, runtime, retraining frequency, training-window policy, and execution status.

## Benchmark methodology

### Forecasting

The quarterly real-GDP series is split chronologically: 80% training and 20% testing. Each forecasting model is fitted on the same training data and predicts exactly the same held-out timestamps.

The primary metric is **sMAPE** (symmetric mean absolute percentage error): lower values are better. Runtime in seconds is also recorded.

### Anomaly detection

Real GDP is converted to quarterly percentage growth. The first 80% is used as clean training history. Five deterministic high-magnitude point anomalies are then injected into the held-out 20% only.

Each detector is trained on the same clean history and evaluated on the same contaminated test period using:

- precision,
- recall,
- F1 score,
- runtime.

Higher F1 is better.

## Why rolling retraining matters

A static holdout benchmark answers, "How well does this model work if trained once?" Production systems usually face a different question: "How well does this model work as new observations arrive and the model is periodically refreshed?"

Merlion's evaluator framework simulates this historical deployment process. It can:

- train an initial model,
- walk forward through unseen observations,
- generate predictions at a configured cadence,
- periodically reset and retrain the model,
- use either the full expanding history or a bounded sliding window,
- evaluate the resulting sequence of predictions against ground truth.

The `production_backtest.py` workflow makes the distinction between static benchmarking and deployment simulation explicit.

## Original synthetic workflow

The synthetic signal combines linear trend, daily and weekly seasonality, Gaussian noise, and deterministic injected anomalies. The first 70% is used for training and the remaining 30% for evaluation.

`DefaultForecaster` produces the forecast and `DefaultDetector` produces anomaly labels. This workflow is intentionally small and useful for learning the Merlion API before moving to the benchmark.

## Why Merlion?

Merlion provides a unified interface for time-series forecasting, anomaly detection, change-point detection, ensembles, AutoML-style model selection, evaluation pipelines, visualization, and distributed execution.

The upstream `salesforce/Merlion` repository was archived in March 2026 and is read-only, so this repository is intended for learning, reproducible experiments, and comparative analysis rather than representing an actively maintained upstream stack.

## References

- Merlion source: https://github.com/salesforce/Merlion
- Merlion documentation: https://opensource.salesforce.com/Merlion/
- Merlion technical report: https://arxiv.org/abs/2109.09265
- statsmodels macrodata dataset: https://www.statsmodels.org/stable/datasets/generated/macrodata.html
