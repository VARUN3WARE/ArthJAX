"""Household agent dynamics."""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr

from arthjax.config import EconomyConfig


def update_households(
    state: Dict,
    company_log_price: jnp.ndarray,
    key: jax.random.PRNGKey,
    cfg: EconomyConfig,
) -> Tuple[Dict, jnp.ndarray]:
    """Behavioral households with type-dependent trading and consumption."""
    key, subkey = jr.split(key)

    nh = cfg.num_households
    nc = cfg.num_companies

    wealth = state["household_wealth"]
    hh_type = state["household_type"]
    income = state["household_income"]
    risk_tol = state["household_risk_tolerance"]
    log_prices = company_log_price
    mean_log_price = jnp.mean(log_prices)
    sentiment = state["sentiment_index"]

    income_adjusted = income * (
        cfg.income_sentiment_base + sentiment * cfg.income_sentiment_scale
    )

    consumption_rate = jnp.array(cfg.consumption_rates)
    consumption_rate_hh = consumption_rate[hh_type.astype(jnp.int32)]
    consumption = wealth * consumption_rate_hh * jnp.clip(0.5 + sentiment, 0.3, 1.5)
    consumption = jnp.minimum(consumption, wealth * 0.4)

    key, subkey = jr.split(key)
    rand_trade = jr.uniform(subkey, (nh,))

    value_signal = -jnp.where(
        hh_type == 0,
        (
            log_prices[jr.randint(jr.fold_in(key, 1), (nh,), 0, nc)]
            - mean_log_price
        )
        / 2.0,
        0.0,
    )

    momentum_signal = jnp.where(
        hh_type == 1, (sentiment - 0.4) * (rand_trade > 0.5), 0.0
    )
    panic_signal = jnp.where(
        hh_type == 2, (0.3 - sentiment) * (rand_trade > 0.6), 0.0
    )
    saver_signal = jnp.where(hh_type == 3, 0.1 * (rand_trade > 0.7), 0.0)

    investment_signal = (
        value_signal + momentum_signal + panic_signal + saver_signal
    ) * risk_tol
    mean_price_change = 0.01
    investment_return = investment_signal * mean_price_change * wealth * 0.2

    new_wealth = wealth + income_adjusted - consumption + investment_return
    key, wkey = jr.split(key)
    wealth_noise = jr.normal(wkey, (nh,)) * 1.5
    new_wealth = new_wealth + wealth_noise
    new_wealth = jnp.clip(new_wealth, cfg.wealth_clip_min, cfg.wealth_clip_max)

    state["household_wealth"] = new_wealth
    state["household_income"] = income_adjusted

    total_demand = jnp.sum(consumption)
    total_demand = jnp.clip(total_demand, cfg.total_demand_min, cfg.total_demand_max)

    return state, total_demand
