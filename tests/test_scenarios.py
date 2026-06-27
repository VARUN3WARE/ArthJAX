"""Scenario preset smoke tests."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from arthjax import EconomyConfig, init_state, simulate
from arthjax.scenarios import SCENARIO_NAMES, apply_scenario_to_state, resolve_scenario
from arthjax.simulation.step import make_step_jit


@pytest.fixture
def tiny_cfg():
    return EconomyConfig(
        num_households=20,
        num_companies=10,
        num_sectors=5,
        default_num_steps=50,
    )


@pytest.mark.parametrize("name", [n for n in SCENARIO_NAMES if n != "baseline"])
def test_scenario_runs_finite(tiny_cfg, name):
    cfg, preset = resolve_scenario(name, tiny_cfg)
    key = jr.PRNGKey(0)
    key, init_key = jr.split(key)
    state = init_state(cfg, init_key)
    state = apply_scenario_to_state(state, preset, cfg)
    step_jit = make_step_jit(cfg)

    _, metrics = simulate(state, 50, key, cfg=cfg, step_jit=step_jit)
    metrics_np = jax.tree.map(np.array, metrics)

    for key_name in ("gdp", "credit_stress", "bad_loan_ratio", "sentiment"):
        assert np.all(np.isfinite(metrics_np[key_name])), key_name


def test_credit_crunch_higher_initial_stress(tiny_cfg):
    cfg_base = tiny_cfg
    cfg_cc, preset = resolve_scenario("credit_crunch", cfg_base)
    key = jr.PRNGKey(1)
    key, init_key = jr.split(key)

    base_state = init_state(cfg_base, init_key)
    cc_state = apply_scenario_to_state(init_state(cfg_cc, init_key), preset, cfg_cc)

    assert float(cc_state["credit_stress_index"]) > float(base_state["credit_stress_index"])
    assert float(cc_state["bank_bad_loans"]) > float(base_state["bank_bad_loans"])
