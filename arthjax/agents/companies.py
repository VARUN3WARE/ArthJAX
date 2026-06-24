"""Company agent dynamics."""

from typing import Dict

import jax
import jax.numpy as jnp
import jax.random as jr

from arthjax.config import EconomyConfig


def update_companies(
    state: Dict,
    total_demand: jnp.ndarray,
    key: jax.random.PRNGKey,
    cfg: EconomyConfig,
) -> Dict:
    """Company financials with order book imbalance, productivity, and leverage."""
    key, subkey = jr.split(key)

    nc = cfg.num_companies
    ns = cfg.num_sectors

    sector = state["company_sector"]
    productivity = state["company_productivity"]
    cash = state["company_cash"]
    debt = state["company_debt"]
    log_price = state["company_log_price"]
    order_imbalance = state["company_order_imbalance"]

    interest_rate = state["interest_rate"]
    credit_stress = state["credit_stress_index"]
    demand_shocks = state["demand_shock_vector"]
    unemployment = state["unemployment"]

    demand_per_sector = total_demand / ns
    base_demand = demand_per_sector / (nc / ns)
    sector_demand_boost = demand_shocks[sector]
    productivity_multiplier = productivity / jnp.mean(productivity)

    revenue = base_demand * (1.0 + sector_demand_boost) * productivity_multiplier
    revenue = revenue * (1.0 - unemployment * 0.5)
    revenue = jnp.clip(revenue, 0.0, 25.0)

    debt_service = debt * (interest_rate * (1.0 + credit_stress * 2.0))
    operating_cost = revenue * (0.5 + (1.0 - productivity) * 0.3)

    profit = revenue - debt_service - operating_cost

    new_cash = cash + profit
    leverage_ratio = debt / (new_cash + debt + 1e-6)

    must_deleverage = leverage_ratio > cfg.leverage_threshold
    deleverage_amount = jnp.where(
        must_deleverage, jnp.minimum(new_cash * 0.2, debt), 0.0
    )

    new_cash = new_cash - deleverage_amount
    new_cash = jnp.clip(new_cash, -5000.0, 10000.0)

    new_debt = debt + deleverage_amount - jnp.minimum(profit * 0.1, debt)
    new_debt = jnp.maximum(new_debt, 0.0)
    new_debt = jnp.clip(new_debt, 0.0, 50000.0)

    profit_momentum = jnp.clip(profit / (revenue + 1e-6), -0.2, 0.2)
    price_impact = order_imbalance * 0.1

    log_return = profit_momentum * 0.012 + price_impact * 0.006
    new_log_price = log_price + log_return
    new_log_price = jnp.clip(
        new_log_price, jnp.log(cfg.log_price_min), jnp.log(cfg.log_price_max)
    )

    new_order_imbalance = order_imbalance * 0.5 + jr.normal(subkey, (nc,)) * 0.05
    new_order_imbalance = jnp.clip(new_order_imbalance, -0.5, 0.5)

    state["company_revenue"] = revenue
    state["company_cash"] = new_cash
    state["company_debt"] = new_debt
    state["company_log_price"] = new_log_price
    state["company_order_imbalance"] = new_order_imbalance

    return state
