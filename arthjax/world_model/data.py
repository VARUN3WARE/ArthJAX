"""World model data collection and normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from arthjax.config import EconomyConfig


def flatten_state(state: dict, cfg: EconomyConfig) -> jnp.ndarray:
    """Flatten economy state to a fixed-size vector for world model I/O."""
    return jnp.concatenate(
        [
            state["household_wealth"],
            state["household_income"],
            state["company_cash"],
            state["company_debt"],
            state["company_log_price"],
            jnp.array(
                [
                    state["interest_rate"],
                    state["inflation"],
                    state["unemployment"],
                    state["sentiment_index"],
                ]
            ),
        ]
    )


def collect_trajectories(
    initial_state: dict,
    step_jit: Callable,
    cfg: EconomyConfig,
    key: jax.random.PRNGKey,
    num_rollouts: int | None = None,
    rollout_length: int | None = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate state transitions for world model training."""
    num_rollouts = num_rollouts or cfg.world_model_num_rollouts
    rollout_length = rollout_length or cfg.world_model_rollout_length
    transitions: List[Tuple[np.ndarray, np.ndarray]] = []

    for _ in range(num_rollouts):
        state = initial_state
        key, subkey = jr.split(key)
        for _ in range(rollout_length):
            state_flat = flatten_state(state, cfg)
            state, _ = step_jit(state, subkey)
            key, subkey = jr.split(subkey)
            transitions.append(
                (np.array(state_flat), np.array(flatten_state(state, cfg)))
            )
    return transitions


@dataclass
class StateNormalizer:
    """Z-score normalizer for world model training."""

    mean: np.ndarray
    std: np.ndarray
    clip: float = 4.0

    @classmethod
    def from_transitions(
        cls,
        transitions: List[Tuple[np.ndarray, np.ndarray]],
        clip: float = 4.0,
    ) -> "StateNormalizer":
        x = np.array([t[0] for t in transitions])
        mean = x.mean(axis=0)
        std = np.maximum(x.std(axis=0), 1e-2)
        return cls(mean=mean, std=std, clip=clip)

    def normalize(self, x) -> jnp.ndarray:
        arr = jnp.asarray(x)
        return (arr - jnp.asarray(self.mean)) / jnp.asarray(self.std)

    def denormalize(self, x_norm) -> np.ndarray:
        arr = np.asarray(x_norm)
        return arr * self.std + self.mean

    def normalize_batch(self, transitions: List[Tuple[np.ndarray, np.ndarray]]):
        x = np.array([t[0] for t in transitions])
        y = np.array([t[1] for t in transitions])
        return self.normalize(x), self.normalize(y)
