"""Simulation loop via lax.scan."""

from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax

from arthjax.config import EconomyConfig, DEFAULT_CONFIG
from arthjax.metrics import compute_metrics
from arthjax.simulation.step import make_step_jit


def simulate(
    state: Dict,
    num_steps: int,
    key: jax.random.PRNGKey,
    cfg: EconomyConfig = DEFAULT_CONFIG,
    step_jit: Callable | None = None,
) -> Tuple[Dict, Dict]:
    """Run simulation vectorized across time using lax.scan."""
    if step_jit is None:
        step_jit = make_step_jit(cfg)

    def scan_fn(carry_state, scan_key):
        new_state, metrics = step_jit(carry_state, scan_key)
        return new_state, metrics

    keys = jr.split(key, num_steps)
    final_state, metrics_stacked = lax.scan(scan_fn, state, keys)

    initial_metrics = compute_metrics(state, cfg)
    metrics_history = jax.tree.map(
        lambda init_m, traj: jnp.concatenate([jnp.expand_dims(init_m, 0), traj], axis=0),
        initial_metrics,
        metrics_stacked,
    )

    return final_state, metrics_history
