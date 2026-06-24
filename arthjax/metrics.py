"""Macro observable metrics."""

from typing import Dict

import jax.numpy as jnp

from arthjax.config import EconomyConfig


def compute_metrics(state: Dict, cfg: EconomyConfig) -> Dict:
    """Calculate macro indicators (all clipped for stability)."""

    def safe_mean(x, default=0.0):
        m = jnp.mean(x)
        return jnp.where(jnp.isnan(m) | jnp.isinf(m), default, m)

    company_prices = jnp.exp(state["company_log_price"])
    stock_market_cap = jnp.sum(company_prices)

    revenue = state["company_revenue"]
    gdp = jnp.sum(revenue)
    gdp = jnp.clip(gdp, cfg.gdp_min, cfg.gdp_max)

    stock_index = safe_mean(company_prices, 100.0)
    stock_index = jnp.clip(stock_index, 1.0, cfg.stock_index_max)

    commodity_prices = jnp.exp(state["commodity_log_price"])
    commodity_index = safe_mean(commodity_prices, 50.0)

    wealth = state["household_wealth"]
    wealth_sorted = jnp.sort(wealth)
    n = cfg.num_households
    cumsum = jnp.cumsum(wealth_sorted)
    total_wealth = cumsum[-1]

    gini = jnp.where(
        total_wealth > 0,
        (2.0 * jnp.sum(jnp.arange(1.0, n + 1) * wealth_sorted))
        / (n * total_wealth)
        - (n + 1) / n,
        0.0,
    )
    gini = jnp.clip(gini, 0.0, 1.0)

    cash = state["company_cash"]
    unhealthy_companies = jnp.sum(cash < 0) / cfg.num_companies

    bad_loan_ratio = jnp.clip(
        state["bank_bad_loans"] / (state["bank_deposits"] + 1e-6), 0.0, 1.0
    )

    return {
        "gdp": gdp,
        "stock_index": stock_index,
        "commodity_index": commodity_index,
        "stock_market_cap": jnp.clip(stock_market_cap, 0.0, 1e6),
        "inflation": jnp.clip(state["inflation"], 0.0, 0.3),
        "unemployment": jnp.clip(state["unemployment"], 0.0, 0.3),
        "interest_rate": jnp.clip(state["interest_rate"], 0.0, 0.2),
        "sentiment": jnp.clip(state["sentiment_index"], 0.0, 1.0),
        "gini_coefficient": gini,
        "bad_loan_ratio": bad_loan_ratio,
        "credit_stress": jnp.clip(state["credit_stress_index"], 0.0, 1.0),
        "unhealthy_companies": jnp.clip(unhealthy_companies, 0.0, 1.0),
        "bank_deposits": jnp.clip(state["bank_deposits"], 0.0, 1e7),
        "bank_leverage": jnp.clip(state["bank_leverage"], 1.0, 20.0),
        "avg_household_wealth": safe_mean(wealth, 100.0),
        "avg_volatility": safe_mean(state["stock_volatility"], 0.15),
        "oil_price_index": jnp.clip(state["oil_price_index"], 0.5, 3.0),
    }
