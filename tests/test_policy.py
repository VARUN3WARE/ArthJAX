"""Policy counterfactual smoke tests."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from arthjax import EconomyConfig, init_state, simulate
from arthjax.policy import POLICY_NAMES, resolve_policy
from arthjax.simulation.step import make_step_jit


@pytest.fixture
def tiny_cfg():
    return EconomyConfig(
        num_households=20,
        num_companies=10,
        num_sectors=5,
        default_num_steps=80,
    )


@pytest.mark.parametrize("name", list(POLICY_NAMES))
def test_policy_preset_runs_finite(tiny_cfg, name):
    cfg, _ = resolve_policy(name, tiny_cfg)
    key = jr.PRNGKey(0)
    state = init_state(cfg, jr.split(key)[0])
    step_jit = make_step_jit(cfg)
    _, metrics = simulate(state, 80, key, cfg=cfg, step_jit=step_jit)
    metrics_np = jax.tree.map(np.array, metrics)
    assert np.all(np.isfinite(metrics_np["gdp"]))
    assert np.all(np.isfinite(metrics_np["interest_rate"]))


def test_same_seed_policy_counterfactual_differs(tiny_cfg):
    """Hawkish vs baseline should diverge on macro path with same init."""
    seed = 42
    init_key = jr.PRNGKey(seed)
    run_key = jr.PRNGKey(seed)

    cfg_base, _ = resolve_policy("baseline", tiny_cfg)
    cfg_hawk, _ = resolve_policy("hawkish", tiny_cfg)

    state = init_state(cfg_base, init_key)
    _, m_base = simulate(
        state, 80, run_key, cfg=cfg_base, step_jit=make_step_jit(cfg_base)
    )
    state2 = init_state(cfg_hawk, init_key)
    _, m_hawk = simulate(
        state2, 80, run_key, cfg=cfg_hawk, step_jit=make_step_jit(cfg_hawk)
    )

    base_rates = np.array(m_base["interest_rate"])
    hawk_rates = np.array(m_hawk["interest_rate"])
    assert not np.allclose(base_rates, hawk_rates, atol=1e-5)
