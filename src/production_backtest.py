"""Production-like Merlion backtesting with rolling retraining.

This module uses Merlion's evaluator classes instead of a single static
train/test split. It compares three deployment policies:

1. train once and never retrain,
2. expanding-window retraining roughly once per year,
3. sliding-window retraining roughly once per year using the latest 10 years.

Forecasting is evaluated on quarterly US real GDP. Anomaly detection is
evaluated on quarterly GDP growth with deterministic anomalies injected only
into the held-out period so that F1/precision/recall are measurable.
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
from merlion.models.forecast.arima import Arima, ArimaConfig
from merlion.utils import TimeSeries

from src.benchmark import inject_growth_anomalies, load_real_gdp, split_frame


ARTIFACT_DIR = Path("artifacts")

# GDP data is quarterly. 100 days reliably reaches the next quarterly timestamp,
# while 365 days gives an intuitive approximately annual retraining policy.
FORECAST_HORIZON = "100D"
ANNUAL_RETRAIN = "365D"
TEN_YEAR_WINDOW = "3650D"


def deployment_policies() -> dict[str, dict[str, str | None]]:
    """Return comparable production retraining policies."""
    return {
        "train_once": {"retrain_freq": None, "train_window": None},
        "expanding_annual": {"retrain_freq": ANNUAL_RETRAIN, "train_window": None},
        "sliding_10y_annual": {
            "retrain_freq": ANNUAL_RETRAIN,
            "train_window": TEN_YEAR_WINDOW,
        },
    }


def run_forecast_backtest(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate ARIMA under multiple live-deployment retraining policies."""
    train_ts = TimeSeries.from_pd(train_df)
    test_ts = TimeSeries.from_pd(test_df)
    rows: list[dict] = []
    predictions: dict[str, pd.Series] = {}

    for policy, params in deployment_policies().items():
        started = perf_counter()
        try:
            model = Arima(ArimaConfig(order=(4, 1, 1)))
            config = ForecastEvaluatorConfig(
                horizon=FORECAST_HORIZON,
                retrain_freq=params["retrain_freq"],
                train_window=params["train_window"],
            )
            evaluator = ForecastEvaluator(model=model, config=config)
            _, prediction = evaluator.get_predict(train_vals=train_ts, test_vals=test_ts)

            smape = float(
                evaluator.evaluate(
                    ground_truth=test_ts,
                    predict=prediction,
                    metric=ForecastMetric.sMAPE,
                )
            )
            rmse = float(
                evaluator.evaluate(
                    ground_truth=test_ts,
                    predict=prediction,
                    metric=ForecastMetric.RMSE,
                )
            )
            elapsed = perf_counter() - started
            predictions[policy] = prediction.to_pd().iloc[:, 0]
            rows.append(
                {
                    "policy": policy,
                    "train_window": params["train_window"] or "expanding/all-history",
                    "retrain_freq": params["retrain_freq"] or "never",
                    "horizon": FORECAST_HORIZON,
                    "sMAPE": smape,
                    "RMSE": rmse,
                    "seconds": elapsed,
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append(
                {
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

    result = pd.DataFrame(rows).sort_values("sMAPE", na_position="last").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train_df.index[-24:], train_df["value"].iloc[-24:], label="train tail", linewidth=2)
    ax.plot(test_df.index, test_df["value"], label="actual", linewidth=2)
    for policy, series in predictions.items():
        ax.plot(series.index, series.values, label=policy, alpha=0.8)
    ax.set_title("Production-like forecast backtest — ARIMA retraining policies")
    ax.set_xlabel("time")
    ax.set_ylabel("US real GDP")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "production_forecast_backtest.png", dpi=160)
    plt.close(fig)

    return result


def run_anomaly_backtest(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate Isolation Forest with Merlion's live TSADEvaluator pipeline."""
    train_ts = TimeSeries.from_pd(train_df)
    test_ts = TimeSeries.from_pd(test_df)
    labels_ts = TimeSeries.from_pd(labels_df)
    rows: list[dict] = []
    predictions: dict[str, pd.Series] = {}

    for policy, params in deployment_policies().items():
        started = perf_counter()
        try:
            model = IsolationForest(IsolationForestConfig(n_estimators=200))
            config = TSADEvaluatorConfig(
                retrain_freq=params["retrain_freq"],
                train_window=params["train_window"],
                cadence="0D",
            )
            evaluator = TSADEvaluator(model=model, config=config)
            _, prediction = evaluator.get_predict(train_vals=train_ts, test_vals=test_ts)

            precision = float(evaluator.evaluate(labels_ts, prediction, TSADMetric.Precision))
            recall = float(evaluator.evaluate(labels_ts, prediction, TSADMetric.Recall))
            f1 = float(evaluator.evaluate(labels_ts, prediction, TSADMetric.F1))
            elapsed = perf_counter() - started

            predictions[policy] = prediction.to_pd().iloc[:, 0].reindex(test_df.index, fill_value=0)
            rows.append(
                {
                    "policy": policy,
                    "train_window": params["train_window"] or "expanding/all-history",
                    "retrain_freq": params["retrain_freq"] or "never",
                    "precision": precision,
                    "recall": recall,
                    "F1": f1,
                    "seconds": elapsed,
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append(
                {
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

    result = pd.DataFrame(rows).sort_values("F1", ascending=False, na_position="last").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test_df.index, test_df["value"], label="GDP growth")
    truth = labels_df["anomaly"].astype(bool)
    ax.scatter(
        test_df.index[truth],
        test_df.loc[truth, "value"],
        marker="x",
        s=85,
        label="true anomaly",
    )
    marker_cycle = ["o", "s", "^"]
    for marker, (policy, series) in zip(marker_cycle, predictions.items()):
        detected = series.astype(bool)
        ax.scatter(
            test_df.index[detected],
            test_df.loc[detected, "value"],
            marker=marker,
            facecolors="none",
            s=90,
            label=f"{policy} detected",
        )
    ax.set_title("Production-like anomaly backtest — Isolation Forest retraining policies")
    ax.set_xlabel("time")
    ax.set_ylabel("quarterly GDP growth (%)")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "production_anomaly_backtest.png", dpi=160)
    plt.close(fig)

    return result


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    real_gdp = load_real_gdp()
    forecast_train, forecast_test = split_frame(real_gdp)
    anomaly_train, anomaly_test, anomaly_labels = inject_growth_anomalies(real_gdp)

    forecast_results = run_forecast_backtest(forecast_train, forecast_test)
    anomaly_results = run_anomaly_backtest(anomaly_train, anomaly_test, anomaly_labels)

    forecast_results.to_csv(ARTIFACT_DIR / "production_forecast_backtest.csv", index=False)
    anomaly_results.to_csv(ARTIFACT_DIR / "production_anomaly_backtest.csv", index=False)

    print("\nProduction forecast backtest (lower is better)")
    print(forecast_results.to_string(index=False))
    print("\nProduction anomaly backtest (higher F1 is better)")
    print(anomaly_results.to_string(index=False))
    print(f"\nArtifacts written to: {ARTIFACT_DIR.resolve()}")


if __name__ == "__main__":
    main()
