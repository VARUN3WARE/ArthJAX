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


def macro_slice(flat: np.ndarray | jnp.ndarray) -> jnp.ndarray:
    """Last 4 components: interest_rate, inflation, unemployment, sentiment."""
    return jnp.asarray(flat)[-4:]


def flatten_macro(state: dict) -> jnp.ndarray:
    """Macro-only vector for world model v2."""
    return jnp.array(
        [
            state["interest_rate"],
            state["inflation"],
            state["unemployment"],
            state["sentiment_index"],
        ]
    )


def embed_macro_in_flat(full_flat: np.ndarray, macro_next: np.ndarray) -> np.ndarray:
    """Replace macro tail of full state vector."""
    out = np.array(full_flat, copy=True)
    out[-4:] = macro_next
    return out


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


def collect_macro_trajectories(
    initial_state: dict,
    step_jit: Callable,
    cfg: EconomyConfig,
    key: jax.random.PRNGKey,
    num_rollouts: int | None = None,
    rollout_length: int | None = None,
    init_state_fn: Callable | None = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Macro-only (4-dim) transitions for world model v2."""
    from arthjax.state import init_state as default_init_state

    init_state_fn = init_state_fn or default_init_state
    num_rollouts = num_rollouts or cfg.world_model_num_rollouts
    rollout_length = rollout_length or cfg.world_model_rollout_length
    transitions: List[Tuple[np.ndarray, np.ndarray]] = []

    for r in range(num_rollouts):
        key, init_key, subkey = jr.split(key, 3)
        state = initial_state if r == 0 else init_state_fn(cfg, init_key)
        for _ in range(rollout_length):
            m0 = np.array(flatten_macro(state))
            state, _ = step_jit(state, subkey)
            key, subkey = jr.split(subkey)
            m1 = np.array(flatten_macro(state))
            transitions.append((m0, m1))
    return transitions


def build_multistep_sequences(
    transitions: List[Tuple[np.ndarray, np.ndarray]],
    horizon: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build (x0, stacked_y) pairs for k-step autoregressive training."""
    if horizon <= 1 or len(transitions) < horizon:
        return [(t[0], t[1]) for t in transitions]

    sequences: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(transitions) - horizon + 1):
        x0 = transitions[i][0]
        future = np.stack([transitions[i + k][1] for k in range(horizon)])
        sequences.append((x0, future))
    return sequences


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
