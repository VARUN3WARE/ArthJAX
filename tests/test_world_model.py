"""World model smoke tests."""

import jax
import jax.random as jr
import numpy as np
import pytest

from arthjax import EconomyConfig, init_state
from arthjax.simulation.step import make_step_jit
from arthjax.world_model.data import flatten_state
from arthjax.world_model.rollout import compare_rollouts, rollout_learned
from arthjax.world_model.train import train_world_model


@pytest.fixture
def tiny_cfg():
    return EconomyConfig(
        num_households=20,
        num_companies=10,
        num_sectors=5,
        num_commodities=4,
        num_currencies=2,
        world_model_epochs=3,
        world_model_num_rollouts=2,
        world_model_rollout_length=20,
        world_model_eval_steps=20,
        world_model_hidden_dim=32,
        world_model_batch_size=32,
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
