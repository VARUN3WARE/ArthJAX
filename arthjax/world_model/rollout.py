"""Learned rollout and evaluation metrics."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from arthjax.config import EconomyConfig
from arthjax.world_model.data import StateNormalizer, flatten_state
from arthjax.world_model.mlp import model_forward


def rollout_learned(
    params: dict,
    initial_flat: np.ndarray,
    normalizer: StateNormalizer,
    cfg: EconomyConfig,
    num_steps: int | None = None,
) -> np.ndarray:
    """Autoregressive rollout in normalized space."""
    num_steps = num_steps or cfg.world_model_eval_steps
    states = [np.array(initial_flat)]
    current_norm = normalizer.normalize(initial_flat)

    for _ in range(num_steps):
        next_norm = model_forward(params, current_norm)
        next_norm = jnp.clip(next_norm, -cfg.world_model_norm_clip, cfg.world_model_norm_clip)
        next_flat = normalizer.denormalize(np.array(next_norm))
        states.append(next_flat)
        current_norm = next_norm

    return np.array(states)


def safe_metric(fn, a, b) -> float:
    v = fn(a, b)
    return float(v) if np.isfinite(v) else 0.0


def compare_rollouts(
    real_trajectory: np.ndarray,
    learned_trajectory: np.ndarray,
) -> dict:
    """Compute full-state and macro-only error metrics."""
    n = min(len(real_trajectory), len(learned_trajectory))
    real_trajectory = real_trajectory[:n]
    learned_trajectory = learned_trajectory[:n]

    macro_idx = real_trajectory.shape[1] - 4
    real_macro = real_trajectory[:, macro_idx:]
    learned_macro = learned_trajectory[:, macro_idx:]

    mse_error = safe_metric(
        lambda a, b: np.mean((a - b) ** 2), real_trajectory, learned_trajectory
    )
    mae_error = safe_metric(
        lambda a, b: np.mean(np.abs(a - b)), real_trajectory, learned_trajectory
    )
    macro_mae = safe_metric(
        lambda a, b: np.mean(np.abs(a - b)), real_macro, learned_macro
    )
    mean_mag = float(np.mean(np.abs(real_trajectory)))
    macro_mag = float(np.mean(np.abs(real_macro)))

    return {
        "mse": mse_error,
        "mae": mae_error,
        "relative_error_pct": (mae_error / (mean_mag + 1e-8)) * 100,
        "macro_mae": macro_mae,
        "macro_relative_error_pct": (macro_mae / (macro_mag + 1e-8)) * 100,
        "num_steps": n,
    }


def collect_real_trajectory(
    initial_state: dict,
    step_jit,
    cfg: EconomyConfig,
    key,
    num_steps: int | None = None,
) -> np.ndarray:
    """Roll out the simulator for comparison."""
    import jax.random as jr

    num_steps = num_steps or cfg.world_model_eval_steps
    real_state = initial_state
    trajectory = []
    for _ in range(num_steps):
        trajectory.append(np.array(flatten_state(real_state, cfg)))
        real_state, _ = step_jit(real_state, key)
        key, _ = jr.split(key)
    return np.array(trajectory)
