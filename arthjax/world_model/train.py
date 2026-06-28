"""World model training orchestration."""

from __future__ import annotations

from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from arthjax.config import EconomyConfig
from arthjax.world_model.data import (
    StateNormalizer,
    build_multistep_sequences,
    collect_macro_trajectories,
    collect_trajectories,
)
from arthjax.world_model.mlp import init_model_params, make_train_step


from arthjax.world_model.mlp import init_model_params, make_train_step
from arthjax.world_model.encoder import (
    init_latent_world_model,
    make_latent_train_step,
)


def _is_latent_params(params: dict) -> bool:
    return isinstance(params, dict) and params.get("latent") is True


def train_world_model(
    initial_state: dict,
    step_jit: Callable,
    cfg: EconomyConfig,
    key: jax.random.PRNGKey,
    transitions: List[Tuple[np.ndarray, np.ndarray]] | None = None,
    macro_only: bool | None = None,
) -> tuple[dict, StateNormalizer, list[float]]:
    """Train MLP (or latent encoder+MLP) on normalized state transitions."""
    macro_only = cfg.world_model_macro_only if macro_only is None else macro_only
    use_latent = cfg.world_model_use_latent and not macro_only

    if transitions is None:
        if macro_only:
            transitions = collect_macro_trajectories(
                initial_state, step_jit, cfg, key
            )
        else:
            transitions = collect_trajectories(initial_state, step_jit, cfg, key)

    normalizer = StateNormalizer.from_transitions(
        transitions, clip=cfg.world_model_norm_clip
    )
    input_dim = transitions[0][0].shape[0]
    output_dim = transitions[0][1].shape[0]

    if use_latent:
        return _train_latent_world_model(
            transitions, normalizer, cfg, key, input_dim
        )

    hidden = cfg.world_model_hidden_dim if not macro_only else max(64, cfg.world_model_hidden_dim // 2)
    key, param_key = jr.split(key)
    params = init_model_params(param_key, input_dim, hidden, output_dim)

    horizon = cfg.world_model_multi_step_horizon if macro_only else 1
    use_multistep = macro_only and horizon > 1
    if use_multistep:
        seqs = build_multistep_sequences(transitions, horizon)
        x_raw = np.array([s[0] for s in seqs])
        y_raw = np.array([s[1] for s in seqs])
        x_norm = normalizer.normalize(x_raw)
        y_norm = np.stack(
            [normalizer.normalize(y_raw[:, h, :]) for h in range(horizon)], axis=1
        )
    else:
        x_norm, y_norm = normalizer.normalize_batch(transitions)

    train_step_jit = make_train_step(cfg, multistep=use_multistep)
    params, losses = _run_training_loop(params, x_norm, y_norm, train_step_jit, cfg)
    return params, normalizer, losses


def _train_latent_world_model(
    transitions: List[Tuple[np.ndarray, np.ndarray]],
    normalizer: StateNormalizer,
    cfg: EconomyConfig,
    key: jax.random.PRNGKey,
    input_dim: int,
) -> tuple[dict, StateNormalizer, list[float]]:
    """Train encoder (684→latent) + latent MLP on full-state transitions."""
    latent_dim = cfg.world_model_latent_dim
    key, init_key = jr.split(key)
    encoder, mlp = init_latent_world_model(
        init_key, input_dim, latent_dim, cfg.world_model_hidden_dim
    )
    x_norm, y_norm = normalizer.normalize_batch(transitions)
    train_step_jit = make_latent_train_step(cfg)
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
            encoder, mlp, loss = train_step_jit(encoder, mlp, xb, yb, lr)
            epoch_loss += float(loss)
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))

    params = {"latent": True, "encoder": encoder, "mlp": mlp}
    return params, normalizer, losses


def _run_training_loop(params, x_norm, y_norm, train_step_jit, cfg) -> tuple[dict, list[float]]:
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
    return params, losses
