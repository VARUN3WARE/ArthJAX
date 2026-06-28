"""World model smoke tests."""

import jax
import jax.random as jr
import numpy as np
import pytest

from arthjax import EconomyConfig, init_state
from arthjax.simulation.step import make_step_jit
from arthjax.world_model.data import flatten_state
from arthjax.world_model.rollout import (
    collect_real_macro_trajectory,
    compare_macro_rollouts,
    compare_rollouts,
    rollout_learned,
)
from arthjax.world_model.train import train_world_model


@pytest.fixture
def tiny_cfg():
    return EconomyConfig(
        num_households=20,
        num_companies=10,
        num_sectors=5,
        num_commodities=4,
        num_currencies=2,
        world_model_epochs=15,
        world_model_num_rollouts=4,
        world_model_rollout_length=40,
        world_model_eval_steps=30,
        world_model_hidden_dim=64,
        world_model_batch_size=32,
        world_model_macro_only=True,
    )


def test_world_model_training_finite(tiny_cfg):
    key = jr.PRNGKey(0)
    key, init_key = jr.split(key)
    state = init_state(tiny_cfg, init_key)
    step_jit = make_step_jit(tiny_cfg)

    params, normalizer, losses = train_world_model(state, step_jit, tiny_cfg, key)

    assert len(losses) == tiny_cfg.world_model_epochs
    assert all(np.isfinite(l) for l in losses)

    flat0 = np.array(flatten_state(state, tiny_cfg))
    learned = rollout_learned(
        params, flat0, normalizer, tiny_cfg, num_steps=tiny_cfg.world_model_eval_steps
    )
    assert np.all(np.isfinite(learned))

    metrics = compare_rollouts(
        np.stack([flat0, flat0]), learned[:2]
    )
    assert np.isfinite(metrics["macro_relative_error_pct"])


def test_world_model_macro_mae_below_threshold(tiny_cfg):
    """Macro-only rollout error should stay below 25% on tiny config."""
    key = jr.PRNGKey(42)
    key, init_key = jr.split(key)
    state = init_state(tiny_cfg, init_key)
    step_jit = make_step_jit(tiny_cfg)

    params, normalizer, _ = train_world_model(
        state, step_jit, tiny_cfg, key, macro_only=True
    )
    real_macro = collect_real_macro_trajectory(
        state,
        step_jit,
        tiny_cfg,
        jr.PRNGKey(tiny_cfg.world_model_eval_seed),
        num_steps=tiny_cfg.world_model_eval_steps,
    )
    learned = rollout_learned(
        params,
        real_macro[0],
        normalizer,
        tiny_cfg,
        num_steps=len(real_macro) - 1,
        macro_only=True,
    )
    metrics = compare_macro_rollouts(real_macro, learned)
    assert metrics["macro_relative_error_pct"] < 25.0, metrics


@pytest.fixture
def latent_cfg():
    return EconomyConfig(
        num_households=20,
        num_companies=10,
        num_sectors=5,
        num_commodities=4,
        num_currencies=2,
        world_model_macro_only=False,
        world_model_use_latent=True,
        world_model_latent_dim=16,
        world_model_epochs=5,
        world_model_num_rollouts=2,
        world_model_rollout_length=20,
        world_model_eval_steps=10,
        world_model_hidden_dim=32,
        world_model_batch_size=16,
    )


def test_latent_world_model_finite(latent_cfg):
    """Full-state latent encoder path produces finite rollouts."""
    key = jr.PRNGKey(7)
    key, init_key = jr.split(key)
    state = init_state(latent_cfg, init_key)
    step_jit = make_step_jit(latent_cfg)

    params, normalizer, losses = train_world_model(
        state, step_jit, latent_cfg, key, macro_only=False
    )
    assert params.get("latent") is True
    assert len(losses) == latent_cfg.world_model_epochs
    assert all(np.isfinite(l) for l in losses)

    from arthjax.world_model.data import flatten_state

    flat0 = np.array(flatten_state(state, latent_cfg))
    learned = rollout_learned(
        params, flat0, normalizer, latent_cfg, num_steps=latent_cfg.world_model_eval_steps
    )
    assert learned.shape[0] == latent_cfg.world_model_eval_steps + 1
    assert np.all(np.isfinite(learned))
