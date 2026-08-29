"""Cross-model walk-forward backtesting across deployment policies.

This module extends ``production_backtest.py`` from a single model per task to a
full ``model x retraining policy`` matrix. Merlion's evaluator classes simulate
historical deployment, including optional periodic retraining and sliding
training windows.

Forecasting models:
- ARIMA
- ETS
- Prophet

Anomaly detectors:
- DefaultDetector
- IsolationForest
- StatThreshold

Deployment policies:
- train_once
- expanding_annual
- sliding_10y_annual

All combinations are fault-tolerant: a failed model/policy pair is recorded in
the output table while the remaining combinations continue.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from merlion.evaluate.anomaly import TSADEvaluator, TSADEvaluatorConfig, TSADMetric
from merlion.evaluate.forecast import ForecastEvaluator, ForecastEvaluatorConfig, ForecastMetric
from merlion.models.anomaly.isolation_forest import IsolationForest, IsolationForestConfig
from merlion.models.anomaly.stat_threshold import StatThreshold, StatThresholdConfig
from merlion.models.defaults import DefaultDetector, DefaultDetectorConfig
from merlion.models.forecast.arima import Arima, ArimaConfig
from merlion.models.forecast.ets import ETS, ETSConfig
from merlion.models.forecast.prophet import Prophet, ProphetConfig
from merlion.utils import TimeSeries

from src.benchmark import inject_growth_anomalies, load_real_gdp, split_frame
from src.production_backtest import (
    ANNUAL_RETRAIN,
    FORECAST_HORIZON,
    TEN_YEAR_WINDOW,
    deployment_policies,
)


ARTIFACT_DIR = Path("artifacts")


def forecast_models() -> dict[str, callable]:
    """Factories for comparable forecasting models."""
    return {
        "ARIMA": lambda: Arima(ArimaConfig(order=(4, 1, 1))),
        "ETS": lambda: ETS(ETSConfig(seasonal_periods=4)),
        "Prophet": lambda: Prophet(
            ProphetConfig(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                uncertainty_samples=100,
            )
        ),
    }


def anomaly_models() -> dict[str, callable]:
    """Factories for comparable anomaly detectors."""
    return {
        "DefaultDetector": lambda: DefaultDetector(DefaultDetectorConfig()),
        "IsolationForest": lambda: IsolationForest(IsolationForestConfig(n_estimators=200)),
        "StatThreshold": lambda: StatThreshold(StatThresholdConfig()),
    }


def run_forecast_matrix(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Run every forecaster under every deployment policy."""
    train_ts = TimeSeries.from_pd(train_df)
    test_ts = TimeSeries.from_pd(test_df)
    rows: list[dict] = []

    for model_name, factory in forecast_models().items():
        for policy, params in deployment_policies().items():
            started = perf_counter()
            try:
                model = factory()
                evaluator = ForecastEvaluator(
                    model=model,
                    config=ForecastEvaluatorConfig(
                        horizon=FORECAST_HORIZON,
                        retrain_freq=params["retrain_freq"],
                        train_window=params["train_window"],
                    ),
                )
                _, prediction = evaluator.get_predict(train_vals=train_ts, test_vals=test_ts)

                smape = float(
                    evaluator.evaluate(test_ts, prediction, ForecastMetric.sMAPE)
                )
                rmse = float(
                    evaluator.evaluate(test_ts, prediction, ForecastMetric.RMSE)
                )
                rows.append(
                    {
                        "model": model_name,
                        "policy": policy,
                        "train_window": params["train_window"] or "expanding/all-history",
                        "retrain_freq": params["retrain_freq"] or "never",
                        "horizon": FORECAST_HORIZON,
                        "sMAPE": smape,
                        "RMSE": rmse,
                        "seconds": perf_counter() - started,
                        "status": "ok",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "model": model_name,
                        "policy": policy,
                        "train_window": params["train_window"] or "expanding/all-history",
                        "retrain_freq": params["retrain_freq"] or "never",
                        "horizon": FORECAST_HORIZON,
                        "sMAPE": np.nan,
                        "RMSE": np.nan,
                        "seconds": perf_counter() - started,
                        "status": f"failed: {type(exc).__name__}: {exc}",
                    }
                )

    result = pd.DataFrame(rows).sort_values(
        ["sMAPE", "RMSE"], na_position="last"
    ).reset_index(drop=True)
    return result


def run_anomaly_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run every anomaly detector under every deployment policy."""
    train_ts = TimeSeries.from_pd(train_df)
    test_ts = TimeSeries.from_pd(test_df)
    labels_ts = TimeSeries.from_pd(labels_df)
    rows: list[dict] = []

    for model_name, factory in anomaly_models().items():
        for policy, params in deployment_policies().items():
            started = perf_counter()
            try:
                model = factory()
                evaluator = TSADEvaluator(
                    model=model,
                    config=TSADEvaluatorConfig(
                        retrain_freq=params["retrain_freq"],
                        train_window=params["train_window"],
                        cadence="0D",
                    ),
                )
                _, prediction = evaluator.get_predict(train_vals=train_ts, test_vals=test_ts)

                precision = float(
                    evaluator.evaluate(labels_ts, prediction, TSADMetric.Precision)
                )
                recall = float(
                    evaluator.evaluate(labels_ts, prediction, TSADMetric.Recall)
                )
                f1 = float(evaluator.evaluate(labels_ts, prediction, TSADMetric.F1))
                rows.append(
                    {
                        "model": model_name,
                        "policy": policy,
                        "train_window": params["train_window"] or "expanding/all-history",
                        "retrain_freq": params["retrain_freq"] or "never",
                        "precision": precision,
                        "recall": recall,
                        "F1": f1,
                        "seconds": perf_counter() - started,
                        "status": "ok",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "model": model_name,
                        "policy": policy,
                        "train_window": params["train_window"] or "expanding/all-history",
                        "retrain_freq": params["retrain_freq"] or "never",
                        "precision": np.nan,
                        "recall": np.nan,
                        "F1": np.nan,
                        "seconds": perf_counter() - started,
                        "status": f"failed: {type(exc).__name__}: {exc}",
                    }
                )

    result = pd.DataFrame(rows).sort_values(
        ["F1", "precision", "recall"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return result


def plot_forecast_heatmap(results: pd.DataFrame) -> None:
    """Plot sMAPE for successful model-policy combinations."""
    pivot = results.pivot(index="model", columns="policy", values="sMAPE")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.set_title("Forecast model × retraining policy — sMAPE (lower is better)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            label = "NA" if pd.isna(value) else f"{value:.2f}"
            ax.text(j, i, label, ha="center", va="center")
    fig.colorbar(image, ax=ax, label="sMAPE")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "forecast_model_policy_matrix.png", dpi=160)
    plt.close(fig)


def plot_anomaly_heatmap(results: pd.DataFrame) -> None:
    """Plot F1 for successful detector-policy combinations."""
    pivot = results.pivot(index="model", columns="policy", values="F1")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.set_title("Anomaly model × retraining policy — F1 (higher is better)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            label = "NA" if pd.isna(value) else f"{value:.2f}"
            ax.text(j, i, label, ha="center", va="center")
    fig.colorbar(image, ax=ax, label="F1")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "anomaly_model_policy_matrix.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    real_gdp = load_real_gdp()
    forecast_train, forecast_test = split_frame(real_gdp)
    anomaly_train, anomaly_test, anomaly_labels = inject_growth_anomalies(real_gdp)

    forecast_results = run_forecast_matrix(forecast_train, forecast_test)
    anomaly_results = run_anomaly_matrix(anomaly_train, anomaly_test, anomaly_labels)

    forecast_results.to_csv(ARTIFACT_DIR / "forecast_model_policy_matrix.csv", index=False)
    anomaly_results.to_csv(ARTIFACT_DIR / "anomaly_model_policy_matrix.csv", index=False)
    plot_forecast_heatmap(forecast_results)
    plot_anomaly_heatmap(anomaly_results)

    print("\nForecast model x policy matrix (lower sMAPE is better)")
    print(forecast_results.to_string(index=False))
    print("\nAnomaly model x policy matrix (higher F1 is better)")
    print(anomaly_results.to_string(index=False))
    print(f"\nArtifacts written to: {ARTIFACT_DIR.resolve()}")


if __name__ == "__main__":
    main()
