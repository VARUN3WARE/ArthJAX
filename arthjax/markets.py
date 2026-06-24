"""Market microstructure dynamics."""

from typing import Dict

import jax
import jax.numpy as jnp
import jax.random as jr

from arthjax.config import EconomyConfig


def update_markets(
    state: Dict,
    key: jax.random.PRNGKey,
    cfg: EconomyConfig,
) -> Dict:
    """Per-stock mean reversion and volatility clustering."""
    key, *subkeys = jr.split(key, 6)

    nc = cfg.num_companies
    ncomm = cfg.num_commodities
    ncurr = cfg.num_currencies

    log_prices = state["company_log_price"]
    volatility = state["stock_volatility"]
    sentiment = state["sentiment_index"]
    inflation = state["inflation"]
    credit_stress = state["credit_stress_index"]

    log_anchor = jnp.log(cfg.log_anchor_price)
    log_min = jnp.log(cfg.log_price_min)
    log_max = jnp.log(cfg.log_price_max)

    per_stock_revert = (log_anchor - log_prices) * cfg.market_revert_rate
    shock_vol = jr.normal(subkeys[0], (nc,)) * 0.04
    volatility_change = shock_vol * 0.08 + (credit_stress - 0.15) * 0.08
    new_volatility = volatility * cfg.vol_decay + jnp.abs(volatility_change) * 0.25
    new_volatility = jnp.clip(new_volatility, cfg.vol_min, cfg.vol_max)

    market_trend = (sentiment - 0.5) * 0.006
    idio_shock = jr.normal(subkeys[1], (nc,)) * new_volatility * 0.45
    new_log_prices = log_prices + per_stock_revert + market_trend + idio_shock
    new_log_prices = jnp.clip(new_log_prices, log_min, log_max)

    comm_log_prices = state["commodity_log_price"]
    oil_shock = jr.normal(subkeys[2], shape=()) * 0.025
    oil_mean_revert = (jnp.log(cfg.commodity_price_base) - comm_log_prices[0]) * 0.04
    new_oil_log = jnp.clip(
        comm_log_prices[0] + oil_shock + oil_mean_revert,
        jnp.log(10.0),
        jnp.log(200.0),
    )

    other_comm_shock = jr.normal(subkeys[3], (ncomm - 1,)) * 0.015
    new_other_comm = comm_log_prices[1:] + inflation * 0.06 + other_comm_shock
    new_comm_log_prices = jnp.concatenate([jnp.array([new_oil_log]), new_other_comm], axis=0)

    curr_rates = state["currency_rates"]
    int_rate_diff = (state["interest_rate"] - cfg.initial_interest_rate) * 0.4
    sentiment_fx = (sentiment - 0.5) * 0.04
    curr_shock = jr.normal(subkeys[4], (ncurr,)) * 0.015
    new_curr_rates = jnp.clip(
        curr_rates * (1.0 + int_rate_diff + sentiment_fx + curr_shock), 0.5, 2.0
    )

    state["company_log_price"] = new_log_prices
    state["stock_volatility"] = new_volatility
    state["commodity_log_price"] = new_comm_log_prices
    state["currency_rates"] = new_curr_rates
    state["oil_price_index"] = jnp.exp(new_oil_log) / cfg.commodity_price_base
    return state
