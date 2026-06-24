"""Structured shock system."""

from typing import Dict

import jax
import jax.numpy as jnp
import jax.random as jr

from arthjax.config import EconomyConfig


def apply_shocks(
    state: Dict,
    key: jax.random.PRNGKey,
    cfg: EconomyConfig,
) -> Dict:
    """Oil, credit, demand, and policy shocks."""
    key, *subkeys = jr.split(key, 10)

    ns = cfg.num_sectors

    has_shock = jr.uniform(subkeys[0]) < cfg.shock_prob
    shock_type = jr.randint(subkeys[1], shape=(), minval=0, maxval=4)
    shock_magnitude = jr.uniform(
        subkeys[2], minval=cfg.shock_magnitude_min, maxval=cfg.shock_magnitude_max
    )

    oil_shock_magnitude = jnp.where(
        has_shock & (shock_type == 0), shock_magnitude, 0.0
    )
    new_oil_idx = state["oil_price_index"] * (1.0 + oil_shock_magnitude)
    new_oil_idx = jnp.clip(new_oil_idx, cfg.oil_index_min, cfg.oil_index_max)

    demand_adj = state["demand_shock_vector"].at[0].add(
        jnp.where(has_shock & (shock_type == 0), oil_shock_magnitude * 0.5, 0.0)
    )

    credit_shock = jnp.where(
        has_shock & (shock_type == 1), shock_magnitude * 0.3, 0.0
    )
    new_credit_stress = state["credit_stress_index"] + credit_shock
    new_credit_stress = jnp.clip(new_credit_stress, 0.0, 1.0)
    rate_from_credit = credit_shock * 0.02

    demand_shock = jnp.where(
        has_shock & (shock_type == 2),
        jr.normal(subkeys[3], (ns,)) * shock_magnitude,
        jnp.zeros(ns),
    )

    policy_rate_shock = jnp.where(
        has_shock & (shock_type == 3),
        (jr.uniform(subkeys[4]) - 0.5) * shock_magnitude * 0.04,
        0.0,
    )

    new_sentiment = state["sentiment_index"]
    new_sentiment = jnp.where(
        has_shock, new_sentiment - shock_magnitude * 0.2, new_sentiment
    )
    new_sentiment = jnp.clip(new_sentiment, 0.1, 0.9)

    new_inflation = state["inflation"]
    new_inflation = jnp.where(
        has_shock & (shock_type == 0),
        new_inflation + oil_shock_magnitude * 0.03,
        new_inflation,
    )
    new_inflation = jnp.clip(new_inflation, 0.0, 0.2)

    new_interest_rate = state["interest_rate"] + rate_from_credit + policy_rate_shock
    new_interest_rate = jnp.clip(new_interest_rate, 0.01, 0.15)

    state["oil_price_index"] = new_oil_idx
    state["credit_stress_index"] = new_credit_stress
    state["demand_shock_vector"] = demand_adj + demand_shock
    state["sentiment_index"] = new_sentiment
    state["inflation"] = new_inflation
    state["interest_rate"] = new_interest_rate

    return state
