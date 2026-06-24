"""Multi-step sector contagion."""

from typing import Dict

import jax.numpy as jnp

from arthjax.config import EconomyConfig


def apply_contagion(state: Dict, cfg: EconomyConfig) -> Dict:
    """Shocks propagate through the sector dependency network."""
    sentiment_history = state["sector_sentiment_history"]
    current_sentiment = sentiment_history[-1]
    dependency = state["sector_dependency"]

    next_sentiment = jnp.dot(dependency.T, current_sentiment)
    new_sentiment = (
        cfg.contagion_momentum * current_sentiment
        + cfg.contagion_propagation * next_sentiment
    )
    new_sentiment = (
        new_sentiment * cfg.contagion_mean_revert + 0.5 * (1.0 - cfg.contagion_mean_revert)
    )
    new_sentiment = jnp.clip(new_sentiment, 0.0, 1.0)

    new_history = jnp.concatenate(
        [sentiment_history[1:], jnp.expand_dims(new_sentiment, 0)], axis=0
    )

    state["sector_sentiment_history"] = new_history
    return state
