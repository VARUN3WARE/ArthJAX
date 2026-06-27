"""Scenario presets — config overrides and initial-state tweaks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Tuple

import jax.numpy as jnp

from arthjax.config import EconomyConfig, DEFAULT_CONFIG

SCENARIO_NAMES: Tuple[str, ...] = ("baseline", "credit_crunch", "oil_shock", "soft_landing")


@dataclass(frozen=True)
class ScenarioPreset:
    """Named preset: optional EconomyConfig overrides + post-init state patches."""

    name: str
    description: str
    config_overrides: Dict = None  # type: ignore[assignment]
    state_patches: Dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "config_overrides", self.config_overrides or {})
        object.__setattr__(self, "state_patches", self.state_patches or {})


def _credit_crunch() -> ScenarioPreset:
    return ScenarioPreset(
        name="credit_crunch",
        description="Elevated credit stress, bad loans, and tighter sentiment at t=0.",
        config_overrides={"shock_prob": 0.04, "initial_sentiment": 0.35},
        state_patches={
            "credit_stress_index": jnp.array(0.35),
            "bank_bad_loans": jnp.array(18000.0),
            "bank_leverage": jnp.array(11.0),
            "sentiment_index": jnp.array(0.35),
            "interest_rate": jnp.array(0.055),
        },
    )


def _oil_shock() -> ScenarioPreset:
    return ScenarioPreset(
        name="oil_shock",
        description="Oil price index elevated; energy-sector demand shock primed.",
        config_overrides={"shock_prob": 0.045},
        state_patches={
            "oil_price_index": jnp.array(1.8),
            "inflation": jnp.array(0.045),
            "demand_shock_vector": None,  # filled at apply time from cfg
        },
    )


def _soft_landing() -> ScenarioPreset:
    return ScenarioPreset(
        name="soft_landing",
        description="Calm initial conditions — low shock rate, healthy banks, mild inflation.",
        config_overrides={
            "shock_prob": 0.012,
            "initial_sentiment": 0.58,
            "initial_inflation": 0.018,
        },
        state_patches={
            "credit_stress_index": jnp.array(0.05),
            "bank_bad_loans": jnp.array(3000.0),
            "bank_leverage": jnp.array(6.5),
            "sentiment_index": jnp.array(0.58),
            "inflation": jnp.array(0.018),
        },
    )


def _baseline() -> ScenarioPreset:
    return ScenarioPreset(
        name="baseline",
        description="Default EconomyConfig with no state patches.",
    )


_PRESETS: Dict[str, ScenarioPreset] = {
    "baseline": _baseline(),
    "credit_crunch": _credit_crunch(),
    "oil_shock": _oil_shock(),
    "soft_landing": _soft_landing(),
}


def get_scenario(name: str) -> ScenarioPreset:
    key = name.lower().strip()
    if key not in _PRESETS:
        raise ValueError(
            f"Unknown scenario {name!r}. Choose from: {', '.join(SCENARIO_NAMES)}"
        )
    return _PRESETS[key]


def resolve_scenario(name: str, base: EconomyConfig = DEFAULT_CONFIG) -> Tuple[EconomyConfig, ScenarioPreset]:
    """Return (merged config, preset) for a scenario name."""
    preset = get_scenario(name)
    if not preset.config_overrides:
        return base, preset
    return replace(base, **preset.config_overrides), preset


def apply_scenario_to_state(state: Dict, preset: ScenarioPreset, cfg: EconomyConfig) -> Dict:
    """Apply preset patches to an initialized state (copy-on-write)."""
    if preset.name == "baseline":
        return state

    new_state = dict(state)
    for key, value in preset.state_patches.items():
        if key == "demand_shock_vector" and value is None:
            boost = jnp.zeros(cfg.num_sectors)
            boost = boost.at[0].set(0.15)
            new_state[key] = state[key] + boost
        elif value is not None:
            new_state[key] = value
    return new_state
