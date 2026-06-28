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


@dataclass
class ComparisonRow:
    """One row in the WM vs baseline comparison table."""

    method: str
    macro_mae: float
    relative_error_pct: float
    elapsed_sec: float
    steps: int


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


def persistence_forecast(series: np.ndarray, horizon: int) -> np.ndarray:
    """Hold last observed value."""
    return np.full(horizon, series[-1])


def _macro_mae(real: np.ndarray, pred: np.ndarray) -> float:
    n = min(len(real), len(pred))
    return float(np.mean(np.abs(real[:n] - pred[:n])))


def _relative_pct(real: np.ndarray, mae: float) -> float:
    mag = float(np.mean(np.abs(real)))
    return (mae / (mag + 1e-8)) * 100


def evaluate_ar1_baseline(
    real_macro: np.ndarray,
    horizon: int | None = None,
) -> BaselineResult:
    """MAE of AR(1) rolled forward averaged across macro columns."""
    horizon = horizon or min(200, len(real_macro) - 1)
    t0 = time.time()
    maes = []
    for col in range(real_macro.shape[1]):
        series = real_macro[:, col]
        train = series[:-horizon]
        actual = series[-horizon:]
        pred = ar1_forecast(train, horizon)
        maes.append(float(np.mean(np.abs(actual - pred))))
    mae = float(np.mean(maes))
    return BaselineResult("AR(1)", mae, horizon, time.time() - t0)


def evaluate_persistence_baseline(
    real_macro: np.ndarray,
    horizon: int | None = None,
) -> BaselineResult:
    """MAE of last-value persistence across macro columns."""
    horizon = horizon or min(200, len(real_macro) - 1)
    t0 = time.time()
    maes = []
    for col in range(real_macro.shape[1]):
        series = real_macro[:, col]
        train = series[:-horizon]
        actual = series[-horizon:]
        pred = persistence_forecast(train, horizon)
        maes.append(float(np.mean(np.abs(actual - pred))))
    mae = float(np.mean(maes))
    return BaselineResult("Persistence", mae, horizon, time.time() - t0)


def build_comparison_table(
    real_macro: np.ndarray,
    full_sim_sec: float,
    sim_steps: int,
    wm_mae: float,
    wm_relative_pct: float,
    wm_rollout_sec: float,
    horizon: int | None = None,
) -> list[ComparisonRow]:
    """Build WM vs AR(1) vs persistence vs full-sim reference table."""
    horizon = horizon or min(50, len(real_macro) - 1)
    eval_slice = real_macro[-horizon:]

    ar1 = evaluate_ar1_baseline(real_macro, horizon=horizon)
    persist = evaluate_persistence_baseline(real_macro, horizon=horizon)

    per_step_sim = full_sim_sec / max(sim_steps, 1)
    sim_horizon_sec = per_step_sim * horizon

    return [
        ComparisonRow(
            "Full simulation (reference)",
            0.0,
            0.0,
            sim_horizon_sec,
            horizon,
        ),
        ComparisonRow(
            "Persistence (last value)",
            persist.mae,
            _relative_pct(eval_slice, persist.mae),
            persist.elapsed_sec,
            horizon,
        ),
        ComparisonRow(
            "AR(1) rolled forward",
            ar1.mae,
            _relative_pct(eval_slice, ar1.mae),
            ar1.elapsed_sec,
            horizon,
        ),
        ComparisonRow(
            "World model (macro)",
            wm_mae,
            wm_relative_pct,
            wm_rollout_sec,
            horizon,
        ),
    ]


def format_comparison_table(rows: list[ComparisonRow]) -> str:
    """Pretty-print comparison table for CLI output."""
    header = f"{'Method':<28} {'Macro MAE':>10} {'Rel %':>8} {'Time (s)':>10} {'Steps':>6}"
    sep = "-" * len(header)
    lines = ["Forecast comparison:", header, sep]
    for r in rows:
        lines.append(
            f"{r.method:<28} {r.macro_mae:>10.6f} {r.relative_error_pct:>7.2f}% "
            f"{r.elapsed_sec:>10.4f} {r.steps:>6d}"
        )
    return "\n".join(lines)


def compare_speed(full_sim_sec: float, wm_rollout_sec: float, steps: int) -> str:
    ratio = full_sim_sec / max(wm_rollout_sec, 1e-6)
    return (
        f"Full sim: {full_sim_sec:.3f}s for {steps} steps | "
        f"WM rollout: {wm_rollout_sec:.3f}s | "
        f"Speedup: {ratio:.1f}x"
    )
