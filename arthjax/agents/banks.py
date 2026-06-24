"""Banking system dynamics."""

from typing import Dict

import jax.numpy as jnp

from arthjax.config import EconomyConfig


def update_banks(state: Dict, cfg: EconomyConfig) -> Dict:
    """Bank dynamics with leverage cycles and credit feedback."""
    cash = state["company_cash"]
    debt = state["company_debt"]
    wealth = state["household_wealth"]

    new_deposits = jnp.sum(wealth) * cfg.savings_rate
    deposits = state["bank_deposits"] + new_deposits * 0.3 - new_deposits * 0.2

    total_debt = jnp.sum(debt)
    default_stress = jnp.sum(cash < cfg.default_cash_threshold) / cfg.num_companies

    bad_loans = state["bank_bad_loans"] * 0.95 + default_stress * deposits * 0.02
    bad_loans = jnp.clip(bad_loans, 0.0, deposits * 0.3)

    good_loans = total_debt * (
        1.0 - jnp.clip(bad_loans / (deposits + 1e-6), 0.0, 1.0)
    )

    interest_income = good_loans * state["interest_rate"] * 0.4
    deposits = deposits + interest_income

    bad_loan_ratio = jnp.clip(bad_loans / (deposits + 1e-6), 0.0, 0.5)
    credit_stress = bad_loan_ratio * 0.9 + state["credit_stress_index"] * 0.1

    capital_ratio = deposits / (good_loans + 1e-6)
    new_leverage = 1.0 / jnp.maximum(capital_ratio, 0.08)

    state["bank_deposits"] = jnp.maximum(deposits, 1000.0)
    state["bank_loans"] = jnp.maximum(good_loans, 0.0)
    state["bank_bad_loans"] = bad_loans
    state["bank_leverage"] = jnp.clip(new_leverage, 2.0, 15.0)
    state["credit_stress_index"] = jnp.clip(credit_stress, 0.0, 1.0)

    return state
