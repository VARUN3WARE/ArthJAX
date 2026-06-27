"""Simple baselines for macro series forecasting."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class BaselineResult:
    name: str
    mae: float
    steps: int
    elapsed_sec: float


def ar1_forecast(series: np.ndarray, horizon: int) -> np.ndarray:
    """One-step AR(1) rolled forward (naive baseline)."""
    if len(series) < 3:
        return np.full(horizon, series[-1])
    x = series[:-1]
    y = series[1:]
    phi = np.corrcoef(x, y)[0, 1] * (np.std(y) / (np.std(x) + 1e-8))
    phi = float(np.clip(phi, -0.99, 0.99))
    c = float(np.mean(y) - phi * np.mean(x))
    preds = []
    last = series[-1]
    for _ in range(horizon):
        last = c + phi * last
        preds.append(last)
    return np.array(preds)


def evaluate_ar1_baseline(
    real_macro: np.ndarray,
    horizon: int | None = None,
) -> BaselineResult:
    """MAE of AR(1) one-step-ahead on macro columns (uses first column as proxy)."""
    horizon = horizon or min(200, len(real_macro) - 1)
    series = real_macro[:, 0]
    train = series[:-horizon]
    actual = series[-horizon:]
    t0 = time.time()
    pred = ar1_forecast(train, horizon)
    mae = float(np.mean(np.abs(actual - pred)))
    return BaselineResult("AR(1)", mae, horizon, time.time() - t0)


def compare_speed(full_sim_sec: float, wm_rollout_sec: float, steps: int) -> str:
    ratio = full_sim_sec / max(wm_rollout_sec, 1e-6)
    return (
        f"Full sim: {full_sim_sec:.3f}s for {steps} steps | "
        f"WM rollout: {wm_rollout_sec:.3f}s | "
        f"Speedup: {ratio:.1f}x"
    )
