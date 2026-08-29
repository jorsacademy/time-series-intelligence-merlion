"""Benchmark multiple Merlion forecasters and anomaly detectors.

The benchmark uses the built-in statsmodels US macroeconomic dataset, so it
requires no network download. Forecasting is evaluated on real GDP. For anomaly
benchmarking, controlled point anomalies are injected into held-out GDP growth
so that precision/recall/F1 can be measured against known labels.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from merlion.evaluate.anomaly import TSADMetric
from merlion.evaluate.forecast import ForecastMetric
from merlion.models.anomaly.isolation_forest import IsolationForest, IsolationForestConfig
from merlion.models.anomaly.stat_threshold import StatThreshold, StatThresholdConfig
from merlion.models.defaults import (
    DefaultDetector,
    DefaultDetectorConfig,
    DefaultForecaster,
    DefaultForecasterConfig,
)
from merlion.models.forecast.arima import Arima, ArimaConfig
from merlion.models.forecast.ets import ETS, ETSConfig
from merlion.models.forecast.prophet import Prophet, ProphetConfig
from merlion.utils import TimeSeries


ARTIFACT_DIR = Path("artifacts")
RANDOM_SEED = 42


def load_real_gdp() -> pd.DataFrame:
    """Load quarterly US real GDP from statsmodels' packaged macro dataset."""
    raw = sm.datasets.macrodata.load_pandas().data.copy()
    quarter = raw["quarter"].astype(int).map({1: 1, 2: 4, 3: 7, 4: 10})
    index = pd.to_datetime(
        {"year": raw["year"].astype(int), "month": quarter, "day": 1}
    )
    frame = pd.DataFrame({"value": raw["realgdp"].to_numpy(dtype=float)}, index=index)
    frame.index.name = "time"
    return frame


def split_frame(frame: pd.DataFrame, train_fraction: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = int(len(frame) * train_fraction)
    return frame.iloc[:split].copy(), frame.iloc[split:].copy()


def inject_growth_anomalies(
    frame: pd.DataFrame, train_fraction: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convert GDP to growth and inject deterministic anomalies only in test data."""
    growth = frame["value"].pct_change().mul(100).dropna().to_frame("value")
    train, test = split_frame(growth, train_fraction)
    labels = pd.DataFrame(0, index=test.index, columns=["anomaly"], dtype=int)

    # Spread anomalies across the held-out horizon and alternate their sign.
    positions = np.linspace(4, len(test) - 5, num=5, dtype=int)
    baseline_scale = max(float(train["value"].std()), 0.5)
    for i, pos in enumerate(positions):
        sign = 1.0 if i % 2 == 0 else -1.0
        test.iloc[pos, test.columns.get_loc("value")] += sign * 6.0 * baseline_scale
        labels.iloc[pos, 0] = 1

    return train, test, labels


def benchmark_forecasters(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    train_ts = TimeSeries.from_pd(train_df)
    test_ts = TimeSeries.from_pd(test_df)

    models = {
        "DefaultForecaster": lambda: DefaultForecaster(DefaultForecasterConfig()),
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

    rows: list[dict] = []
    forecasts: dict[str, pd.Series] = {}

    for name, factory in models.items():
        started = perf_counter()
        try:
            model = factory()
            model.train(train_data=train_ts)
            prediction, _ = model.forecast(time_stamps=test_ts.time_stamps)
            elapsed = perf_counter() - started
            smape = float(ForecastMetric.sMAPE.value(ground_truth=test_ts, predict=prediction))
            forecasts[name] = prediction.to_pd().iloc[:, 0]
            rows.append({"model": name, "sMAPE": smape, "seconds": elapsed, "status": "ok"})
        except Exception as exc:  # benchmark should continue if one optional model fails
            rows.append(
                {
                    "model": name,
                    "sMAPE": np.nan,
                    "seconds": perf_counter() - started,
                    "status": f"failed: {type(exc).__name__}: {exc}",
                }
            )

    result = pd.DataFrame(rows).sort_values("sMAPE", na_position="last").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train_df.index[-24:], train_df["value"].iloc[-24:], label="train tail", linewidth=2)
    ax.plot(test_df.index, test_df["value"], label="actual", linewidth=2)
    for name, series in forecasts.items():
        ax.plot(series.index, series.values, label=name, alpha=0.8)
    ax.set_title("Merlion forecasting benchmark — US real GDP")
    ax.set_xlabel("time")
    ax.set_ylabel("real GDP")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "forecast_benchmark.png", dpi=160)
    plt.close(fig)

    return result


def benchmark_detectors(
    train_df: pd.DataFrame, test_df: pd.DataFrame, labels_df: pd.DataFrame
) -> pd.DataFrame:
    train_ts = TimeSeries.from_pd(train_df)
    test_ts = TimeSeries.from_pd(test_df)
    labels_ts = TimeSeries.from_pd(labels_df)

    models = {
        "DefaultDetector": lambda: DefaultDetector(DefaultDetectorConfig()),
        "IsolationForest": lambda: IsolationForest(IsolationForestConfig(n_estimators=200)),
        "StatThreshold": lambda: StatThreshold(StatThresholdConfig()),
    }

    rows: list[dict] = []
    predictions: dict[str, pd.Series] = {}

    for name, factory in models.items():
        started = perf_counter()
        try:
            model = factory()
            model.train(train_data=train_ts)
            predicted = model.get_anomaly_label(time_series=test_ts)
            elapsed = perf_counter() - started
            precision = float(TSADMetric.Precision.value(ground_truth=labels_ts, predict=predicted))
            recall = float(TSADMetric.Recall.value(ground_truth=labels_ts, predict=predicted))
            f1 = float(TSADMetric.F1.value(ground_truth=labels_ts, predict=predicted))
            predictions[name] = predicted.to_pd().iloc[:, 0].reindex(test_df.index, fill_value=0)
            rows.append(
                {
                    "model": name,
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
                    "model": name,
                    "precision": np.nan,
                    "recall": np.nan,
                    "F1": np.nan,
                    "seconds": perf_counter() - started,
                    "status": f"failed: {type(exc).__name__}: {exc}",
                }
            )

    result = pd.DataFrame(rows).sort_values("F1", ascending=False, na_position="last").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test_df.index, test_df["value"], label="GDP growth (with injected anomalies)")
    truth = labels_df["anomaly"].astype(bool)
    ax.scatter(test_df.index[truth], test_df.loc[truth, "value"], marker="x", s=80, label="true anomaly")

    marker_cycle = ["o", "s", "^"]
    for marker, (name, series) in zip(marker_cycle, predictions.items()):
        detected = series.astype(bool)
        ax.scatter(
            test_df.index[detected],
            test_df.loc[detected, "value"],
            marker=marker,
            facecolors="none",
            s=90,
            label=f"{name} detected",
        )

    ax.set_title("Merlion anomaly-detection benchmark — real GDP growth + controlled anomalies")
    ax.set_xlabel("time")
    ax.set_ylabel("quarterly growth (%)")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "anomaly_benchmark.png", dpi=160)
    plt.close(fig)

    return result


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    real_gdp = load_real_gdp()
    forecast_train, forecast_test = split_frame(real_gdp)
    anomaly_train, anomaly_test, anomaly_labels = inject_growth_anomalies(real_gdp)

    forecast_results = benchmark_forecasters(forecast_train, forecast_test)
    anomaly_results = benchmark_detectors(anomaly_train, anomaly_test, anomaly_labels)

    forecast_results.to_csv(ARTIFACT_DIR / "forecast_benchmark.csv", index=False)
    anomaly_results.to_csv(ARTIFACT_DIR / "anomaly_benchmark.csv", index=False)

    print("\nForecast benchmark (lower sMAPE is better)")
    print(forecast_results.to_string(index=False))
    print("\nAnomaly benchmark (higher F1 is better)")
    print(anomaly_results.to_string(index=False))
    print(f"\nArtifacts written to: {ARTIFACT_DIR.resolve()}")


if __name__ == "__main__":
    main()
