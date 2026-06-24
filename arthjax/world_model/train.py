"""World model training orchestration."""

from __future__ import annotations

from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from arthjax.config import EconomyConfig
from arthjax.world_model.data import StateNormalizer, collect_trajectories
from arthjax.world_model.mlp import init_model_params, make_train_step


def train_world_model(
    initial_state: dict,
    step_jit: Callable,
    cfg: EconomyConfig,
    key: jax.random.PRNGKey,
    transitions: List[Tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[dict, StateNormalizer, list[float]]:
    """Train MLP on normalized state transitions."""
    if transitions is None:
        transitions = collect_trajectories(initial_state, step_jit, cfg, key)

    normalizer = StateNormalizer.from_transitions(
        transitions, clip=cfg.world_model_norm_clip
    )
    x_norm, y_norm = normalizer.normalize_batch(transitions)
    input_dim = x_norm.shape[1]
    output_dim = y_norm.shape[1]

    key, param_key = jr.split(key)
    params = init_model_params(
        param_key, input_dim, cfg.world_model_hidden_dim, output_dim
    )
    train_step_jit = make_train_step(cfg)

    losses: list[float] = []
    batch_size = cfg.world_model_batch_size
    lr = cfg.world_model_lr

    for _epoch in range(cfg.world_model_epochs):
        epoch_loss, n_batches = 0.0, 0
        perm = np.random.permutation(len(x_norm))
        for i in range(0, len(x_norm), batch_size):
            idx = perm[i : i + batch_size]
            xb = jnp.array(x_norm[idx])
            yb = jnp.array(y_norm[idx])
            params, loss = train_step_jit(params, xb, yb, lr)
            epoch_loss += float(loss)
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))

    return params, normalizer, losses
