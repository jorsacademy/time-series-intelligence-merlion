"""End-to-end forecasting and anomaly-detection demo with Merlion."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from merlion.evaluate.anomaly import TSADMetric
from merlion.evaluate.forecast import ForecastMetric
from merlion.models.defaults import (
    DefaultDetector,
    DefaultDetectorConfig,
    DefaultForecaster,
    DefaultForecasterConfig,
)
from merlion.utils import TimeSeries

from src.data import make_dataset


ARTIFACT_DIR = Path("artifacts")


def _first_column(frame: pd.DataFrame) -> pd.Series:
    """Return the first data column from a Merlion-to-pandas conversion."""
    return frame.iloc[:, 0]


def run_forecasting(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float]:
    """Train Merlion's default forecaster and evaluate it on the test horizon."""
    train_ts = TimeSeries.from_pd(train_df)
    test_ts = TimeSeries.from_pd(test_df)

    model = DefaultForecaster(DefaultForecasterConfig())
    model.train(train_data=train_ts)
    prediction, _ = model.forecast(time_stamps=test_ts.time_stamps)

    smape = float(ForecastMetric.sMAPE.value(ground_truth=test_ts, predict=prediction))

    pred_series = _first_column(prediction.to_pd())
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(train_df.index[-7 * 24 :], train_df["value"].iloc[-7 * 24 :], label="train tail")
    ax.plot(test_df.index, test_df["value"], label="actual")
    ax.plot(pred_series.index, pred_series.values, label="forecast")
    ax.set_title(f"Merlion Forecast — sMAPE: {smape:.2f}")
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "forecast.png", dpi=150)
    plt.close(fig)

    return {"forecast_smape": smape}


def run_anomaly_detection(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_labels_df: pd.DataFrame,
) -> dict[str, float]:
    """Train Merlion's default detector and score held-out observations."""
    train_ts = TimeSeries.from_pd(train_df)
    test_ts = TimeSeries.from_pd(test_df)
    labels_ts = TimeSeries.from_pd(test_labels_df)

    model = DefaultDetector(DefaultDetectorConfig())
    model.train(train_data=train_ts)
    predicted_labels = model.get_anomaly_label(time_series=test_ts)

    precision = float(TSADMetric.Precision.value(ground_truth=labels_ts, predict=predicted_labels))
    recall = float(TSADMetric.Recall.value(ground_truth=labels_ts, predict=predicted_labels))
    f1 = float(TSADMetric.F1.value(ground_truth=labels_ts, predict=predicted_labels))

    predicted = _first_column(predicted_labels.to_pd()).reindex(test_df.index, fill_value=0)
    true_labels = test_labels_df["anomaly"].astype(bool)
    detected = predicted.astype(bool)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(test_df.index, test_df["value"], label="observed")
    ax.scatter(
        test_df.index[true_labels],
        test_df.loc[true_labels, "value"],
        marker="x",
        s=70,
        label="true anomaly",
    )
    ax.scatter(
        test_df.index[detected],
        test_df.loc[detected, "value"],
        facecolors="none",
        s=90,
        label="detected",
    )
    ax.set_title(f"Merlion Anomaly Detection — F1: {f1:.2f}")
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "anomaly_detection.png", dpi=150)
    plt.close(fig)

    return {
        "anomaly_precision": precision,
        "anomaly_recall": recall,
        "anomaly_f1": f1,
    }


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = make_dataset()
    split = dataset.split_index

    train_df = dataset.values.iloc[:split]
    test_df = dataset.values.iloc[split:]
    test_labels_df = dataset.labels.iloc[split:]

    metrics = {}
    metrics.update(run_forecasting(train_df, test_df))
    metrics.update(run_anomaly_detection(train_df, test_df, test_labels_df))

    metrics_frame = pd.DataFrame(
        [{"metric": name, "value": value} for name, value in metrics.items()]
    )
    metrics_frame.to_csv(ARTIFACT_DIR / "metrics.csv", index=False)

    print("\nMerlion demo complete")
    print(metrics_frame.to_string(index=False))
    print(f"\nArtifacts written to: {ARTIFACT_DIR.resolve()}")


if __name__ == "__main__":
    main()
