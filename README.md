# Time Series Intelligence with Merlion

A compact, reproducible project demonstrating **forecasting**, **anomaly detection**, and **benchmark-style evaluation** with Salesforce's [Merlion](https://github.com/salesforce/Merlion) time-series library.

The project intentionally uses synthetic data so it can run end-to-end without downloading an external dataset. The generated series contains trend, seasonality, noise, and injected point anomalies.

## What this repository demonstrates

- Merlion `TimeSeries` conversion from pandas data
- Forecasting with `DefaultForecaster`
- Anomaly detection with `DefaultDetector`
- Forecast evaluation with sMAPE
- Anomaly evaluation with precision, recall, and F1
- Reproducible synthetic time-series generation
- PNG result plots and a CSV metrics summary

## Project structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
└── src
    ├── __init__.py
    ├── data.py
    └── run_demo.py
```

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Merlion is published on PyPI as `salesforce-merlion`.

## Run the demo

```bash
python -m src.run_demo
```

The script creates an `artifacts/` directory containing:

- `forecast.png` — actual values vs. Merlion forecast
- `anomaly_detection.png` — observed series with detected and true anomalies
- `metrics.csv` — forecasting and anomaly-detection metrics

## Method

### Synthetic signal

The signal combines:

- linear trend,
- daily and weekly seasonality,
- Gaussian noise,
- deterministic injected anomalies in the held-out period.

The first 70% of observations are used for model training and the remaining 30% for evaluation.

### Forecasting

`DefaultForecaster` provides Merlion's practical default forecasting configuration. The model is trained on the clean training split and asked to forecast the timestamps in the test split.

Forecast quality is summarized with **sMAPE** (symmetric mean absolute percentage error). Lower is better.

### Anomaly detection

`DefaultDetector` is trained on the clean training split. It then produces anomaly labels for the held-out test series, which contains injected anomalies.

Detection quality is summarized with:

- Precision
- Recall
- F1 score

## Why Merlion?

Merlion provides a unified API for multiple time-series tasks, including forecasting, anomaly detection, change-point detection, model ensembles, AutoML-style model selection, evaluation pipelines, and distributed execution support.

This repository focuses on the two most common entry points—forecasting and anomaly detection—while keeping the example small enough to inspect and modify.

## Notes

The upstream `salesforce/Merlion` repository is archived, but the package and source remain useful for learning and reproducible experiments. For production use, pin dependencies and validate the stack on the target Python/platform combination.

## References

- Merlion source: https://github.com/salesforce/Merlion
- Merlion documentation: https://opensource.salesforce.com/Merlion/
- Technical report: https://arxiv.org/abs/2109.09265
