"""Economy state schema and initialization."""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr

from arthjax.config import EconomyConfig, DEFAULT_CONFIG

STATE_KEYS: Tuple[str, ...] = (
    "household_wealth",
    "household_type",
    "household_income",
    "household_risk_tolerance",
    "company_cash",
    "company_debt",
    "company_sector",
    "company_productivity",
    "company_log_price",
    "company_order_imbalance",
    "company_revenue",
    "bank_deposits",
    "bank_loans",
    "bank_bad_loans",
    "bank_leverage",
    "commodity_log_price",
    "currency_rates",
    "stock_volatility",
    "interest_rate",
    "inflation",
    "unemployment",
    "sentiment_index",
    "sector_dependency",
    "sector_sentiment_history",
    "oil_price_index",
    "credit_stress_index",
    "demand_shock_vector",
    "time_step",
)

METRIC_KEYS: Tuple[str, ...] = (
    "gdp",
    "stock_index",
    "commodity_index",
    "stock_market_cap",
    "inflation",
    "unemployment",
    "interest_rate",
    "sentiment",
    "gini_coefficient",
    "bad_loan_ratio",
    "credit_stress",
    "unhealthy_companies",
    "bank_deposits",
    "bank_leverage",
    "avg_household_wealth",
    "avg_volatility",
    "oil_price_index",
)


def init_state(
    cfg: EconomyConfig,
    key: jax.random.PRNGKey,
) -> Dict:
    """Initialize economy with behavioral agents, order imbalances, and contagion buffers."""
    key, *subkeys = jr.split(key, 50)

    nh = cfg.num_households
    nc = cfg.num_companies
    ns = cfg.num_sectors
    ncomm = cfg.num_commodities
    ncurr = cfg.num_currencies
    nat = cfg.num_agent_types

    household_wealth = jnp.full((nh,), cfg.household_wealth_base) + jr.normal(
        subkeys[0], (nh,)
    ) * cfg.household_wealth_noise
    household_wealth = jnp.clip(
        household_wealth, cfg.household_wealth_min, cfg.household_wealth_max
    )

    household_type = jr.randint(subkeys[1], (nh,), 0, nat).astype(jnp.float32)

    household_income = jnp.full((nh,), cfg.household_income_base) + jr.normal(
        subkeys[2], (nh,)
    ) * cfg.household_income_noise
    household_income = jnp.clip(
        household_income, cfg.household_income_min, cfg.household_income_max
    )

    base_risk = jnp.array(cfg.household_risk_by_type)
    household_risk_tolerance = base_risk[household_type.astype(jnp.int32)] + jr.normal(
        subkeys[3], (nh,)
    ) * 0.1
    household_risk_tolerance = jnp.clip(household_risk_tolerance, 0.01, 0.99)

    company_cash = jnp.full((nc,), cfg.company_cash_base) + jr.normal(
        subkeys[4], (nc,)
    ) * cfg.company_cash_noise
    company_cash = jnp.clip(company_cash, 50.0, 5000.0)

    company_debt = jnp.full((nc,), cfg.company_debt_base) + jr.normal(
        subkeys[5], (nc,)
    ) * cfg.company_debt_noise
    company_debt = jnp.maximum(company_debt, 0.0)

    company_sector = jr.randint(subkeys[6], (nc,), 0, ns).astype(jnp.int32)

    company_productivity = jnp.exp(jr.normal(subkeys[7], (nc,)) * 0.5 - 0.25)
    company_productivity = jnp.clip(company_productivity, 0.5, 3.0)

    company_stock_price = jnp.full((nc,), cfg.company_stock_price_base) + jr.normal(
        subkeys[8], (nc,)
    ) * cfg.company_stock_price_noise
    company_stock_price = jnp.clip(company_stock_price, 10.0, 1000.0)
    company_log_price = jnp.log(company_stock_price)

    company_order_imbalance = jr.uniform(subkeys[9], (nc,)) * 0.3 - 0.15

    bank_deposits = jnp.array(cfg.bank_deposits_init)
    bank_loans = jnp.array(cfg.bank_loans_init)
    bank_bad_loans = jnp.array(cfg.bank_bad_loans_init)
    bank_leverage = jnp.array(cfg.bank_leverage_init)

    commodity_prices = jnp.full((ncomm,), cfg.commodity_price_base) + jr.normal(
        subkeys[10], (ncomm,)
    ) * 10
    commodity_prices = jnp.clip(commodity_prices, 1.0, 200.0)
    commodity_log_price = jnp.log(commodity_prices)

    currency_rates = jnp.full((ncurr,), 1.0) + jr.normal(subkeys[11], (ncurr,)) * 0.1
    currency_rates = jnp.clip(currency_rates, 0.5, 2.0)

    interest_rate = jnp.array(cfg.initial_interest_rate)
    inflation = jnp.array(cfg.initial_inflation)
    unemployment = jnp.array(cfg.initial_unemployment)
    sentiment_index = jnp.array(cfg.initial_sentiment)

    stock_volatility = jnp.full((nc,), cfg.stock_volatility_init)

    sector_dependency = jr.uniform(subkeys[12], (ns, ns)) * 0.3
    sector_dependency = sector_dependency / (
        jnp.sum(sector_dependency, axis=1, keepdims=True) + 1e-8
    )

    sector_sentiment_history = jnp.full((cfg.contagion_steps, ns), 0.5)

    oil_price_index = jnp.array(1.0)
    credit_stress_index = jnp.array(0.0)
    demand_shock_vector = jr.normal(subkeys[13], (ns,)) * 0.05

    return {
        "household_wealth": household_wealth,
        "household_type": household_type,
        "household_income": household_income,
        "household_risk_tolerance": household_risk_tolerance,
        "company_cash": company_cash,
        "company_debt": company_debt,
        "company_sector": company_sector,
        "company_productivity": company_productivity,
        "company_log_price": company_log_price,
        "company_order_imbalance": company_order_imbalance,
        "company_revenue": jnp.full((nc,), cfg.company_initial_revenue),
        "bank_deposits": bank_deposits,
        "bank_loans": bank_loans,
        "bank_bad_loans": bank_bad_loans,
        "bank_leverage": bank_leverage,
        "commodity_log_price": commodity_log_price,
        "currency_rates": currency_rates,
        "stock_volatility": stock_volatility,
        "interest_rate": interest_rate,
        "inflation": inflation,
        "unemployment": unemployment,
        "sentiment_index": sentiment_index,
        "sector_dependency": sector_dependency,
        "sector_sentiment_history": sector_sentiment_history,
        "oil_price_index": oil_price_index,
        "credit_stress_index": credit_stress_index,
        "demand_shock_vector": demand_shock_vector,
        "time_step": jnp.array(0, dtype=jnp.int32),
    }
