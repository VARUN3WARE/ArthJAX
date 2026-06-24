"""Simulation smoke tests."""

import jax
import jax.numpy as jnp
import pytest

from arthjax import EconomyConfig, init_state, simulate
from arthjax.state import METRIC_KEYS, STATE_KEYS
from arthjax.simulation.step import make_step_jit


@pytest.fixture
def tiny_cfg():
    return EconomyConfig(
        num_households=20,
        num_companies=10,
        num_sectors=5,
        num_commodities=4,
        num_currencies=2,
        default_num_steps=50,
    )


def test_init_state_shapes(tiny_cfg):
    key = jax.random.PRNGKey(0)
    state = init_state(tiny_cfg, key)
    assert set(state.keys()) == set(STATE_KEYS)
    assert state["household_wealth"].shape == (tiny_cfg.num_households,)
    assert state["company_cash"].shape == (tiny_cfg.num_companies,)
    assert state["sector_dependency"].shape == (
        tiny_cfg.num_sectors,
        tiny_cfg.num_sectors,
    )


def test_step_no_nan(tiny_cfg):
    key = jax.random.PRNGKey(1)
    state = init_state(tiny_cfg, key)
    key, sim_key = jax.random.split(key)
    _, metrics_history = simulate(state, 100, sim_key, cfg=tiny_cfg)
    for key_name in METRIC_KEYS:
        assert jnp.all(jnp.isfinite(metrics_history[key_name])), key_name


def test_simulate_output_length(tiny_cfg):
    num_steps = 50
    key = jax.random.PRNGKey(2)
    state = init_state(tiny_cfg, key)
    key, sim_key = jax.random.split(key)
    _, metrics_history = simulate(state, num_steps, sim_key, cfg=tiny_cfg)
    assert metrics_history["gdp"].shape[0] == num_steps + 1
