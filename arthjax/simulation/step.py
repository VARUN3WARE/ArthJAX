"""Single simulation step."""

from typing import Callable, Dict, Tuple

import jax
import jax.random as jr
from jax import jit

from arthjax.agents.banks import update_banks
from arthjax.agents.companies import update_companies
from arthjax.agents.households import update_households
from arthjax.config import EconomyConfig
from arthjax.contagion import apply_contagion
from arthjax.macro import apply_macro_feedback
from arthjax.markets import update_markets
from arthjax.metrics import compute_metrics
from arthjax.shocks import apply_shocks


def make_step(cfg: EconomyConfig) -> Callable:
    """Build a simulation step with cfg baked into the closure."""

    def step(state: Dict, key: jax.random.PRNGKey) -> Tuple[Dict, Dict]:
        key, *subkeys = jr.split(key, 10)

        state, total_demand = update_households(
            state, state["company_log_price"], subkeys[0], cfg
        )
        state = update_companies(state, total_demand, subkeys[1], cfg)
        state = update_banks(state, cfg)
        state = update_markets(state, subkeys[2], cfg)
        state = apply_shocks(state, subkeys[3], cfg)
        state = apply_contagion(state, cfg)
        state = apply_macro_feedback(state, cfg)
        metrics = compute_metrics(state, cfg)
        state["time_step"] = state["time_step"] + 1
        return state, metrics

    return step


def make_step_jit(cfg: EconomyConfig) -> Callable:
    """JIT-compiled simulation step for a fixed config."""
    return jit(make_step(cfg))
