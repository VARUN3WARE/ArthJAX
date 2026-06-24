"""Macro feedback: Phillips curve and Taylor rule."""

from typing import Dict

import jax.numpy as jnp

from arthjax.config import EconomyConfig


def apply_macro_feedback(state: Dict, cfg: EconomyConfig) -> Dict:
    """Phillips curve + Taylor rule with business-cycle dynamics."""
    rates = state["interest_rate"]
    inflation = state["inflation"]
    unemployment = state["unemployment"]
    credit_stress = state["credit_stress_index"]
    t = state["time_step"].astype(jnp.float32)

    real_rate = rates - inflation
    growth = cfg.growth_base - real_rate * 1.5 - credit_stress * 0.03
    growth = jnp.clip(growth, -0.06, 0.06)

    cycle = (
        jnp.sin(t * cfg.cycle_freq_1) * 0.003
        + jnp.sin(t * cfg.cycle_freq_2) * 0.002
    )

    nairu = cfg.nairu
    new_unemployment = (
        unemployment * 0.92 + (nairu - growth * 2.0) * 0.08 + cycle
    )
    new_unemployment = jnp.clip(
        new_unemployment, cfg.unemployment_min, cfg.unemployment_max
    )

    phillips_effect = (
        -(new_unemployment - nairu) * 0.5
        + inflation * 0.88
        + credit_stress * 0.05
    )
    new_inflation = inflation + phillips_effect * 0.006 + cycle * 0.5
    new_inflation = jnp.clip(new_inflation, 0.008, 0.10)

    inflation_gap = new_inflation - cfg.inflation_target
    unemployment_gap = new_unemployment - nairu
    policy_adjustment = inflation_gap * 1.0 + unemployment_gap * -0.35
    new_rates = rates + policy_adjustment * 0.012
    new_rates = jnp.clip(new_rates, cfg.rate_min, cfg.rate_max)

    state["unemployment"] = new_unemployment
    state["inflation"] = new_inflation
    state["interest_rate"] = new_rates
    return state
