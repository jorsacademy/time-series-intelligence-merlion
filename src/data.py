"""Synthetic dataset utilities."""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Dataset:
    values: pd.DataFrame
    labels: pd.DataFrame
    split_index: int


def make_dataset(n_points: int = 1008, train_fraction: float = 0.70, seed: int = 7) -> Dataset:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=n_points, freq="h")
    t = np.arange(n_points, dtype=float)

    values = (
        18.0
        + 0.012 * t
        + 2.8 * np.sin(2 * np.pi * t / 24)
        + 1.4 * np.sin(2 * np.pi * t / (24 * 7))
        + rng.normal(0.0, 0.55, n_points)
    )

    split_index = int(n_points * train_fraction)
    labels = np.zeros(n_points, dtype=int)
    offsets = [24, 73, 121, 166, 231]
    magnitudes = [8.5, -7.5, 10.0, -9.0, 7.0]

    for offset, magnitude in zip(offsets, magnitudes):
        i = split_index + offset
        if i < n_points:
            values[i] += magnitude
            labels[i] = 1

    frame = pd.DataFrame({"value": values}, index=index)
    label_frame = pd.DataFrame({"anomaly": labels}, index=index)
    return Dataset(frame, label_frame, split_index)
