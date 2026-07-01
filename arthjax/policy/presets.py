"""Taylor-rule policy presets for counterfactual runs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Tuple

from arthjax.config import EconomyConfig, DEFAULT_CONFIG

POLICY_NAMES: Tuple[str, ...] = ("baseline", "hawkish", "dovish")


@dataclass(frozen=True)
class PolicyPreset:
    name: str
    description: str
    config_overrides: Dict


def _baseline() -> PolicyPreset:
    return PolicyPreset(
        name="baseline",
        description="Default Taylor-rule coefficients from EconomyConfig.",
        config_overrides={},
    )


def _hawkish() -> PolicyPreset:
    return PolicyPreset(
        name="hawkish",
        description="Lower inflation target, stronger rate response to inflation gaps.",
        config_overrides={
            "inflation_target": 0.02,
            "taylor_inflation_coef": 1.45,
            "taylor_unemployment_coef": -0.55,
            "taylor_rate_step": 0.014,
        },
    )


def _dovish() -> PolicyPreset:
    return PolicyPreset(
        name="dovish",
        description="Higher inflation target, softer rate hikes, more unemployment weight.",
        config_overrides={
            "inflation_target": 0.032,
            "taylor_inflation_coef": 0.75,
            "taylor_unemployment_coef": -0.22,
            "taylor_rate_step": 0.009,
        },
    )


_PRESETS: Dict[str, PolicyPreset] = {
    "baseline": _baseline(),
    "hawkish": _hawkish(),
    "dovish": _dovish(),
}


def get_policy(name: str) -> PolicyPreset:
    key = name.lower().strip()
    if key not in _PRESETS:
        raise ValueError(
            f"Unknown policy {name!r}. Choose from: {', '.join(POLICY_NAMES)}"
        )
    return _PRESETS[key]


def resolve_policy(
    name: str, base: EconomyConfig = DEFAULT_CONFIG
) -> Tuple[EconomyConfig, PolicyPreset]:
    """Return (merged config, preset) for a policy name."""
    preset = get_policy(name)
    if not preset.config_overrides:
        return base, preset
    return replace(base, **preset.config_overrides), preset
