"""Economy-wide parameters (single source of truth for v0.2 extraction)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EconomyConfig:
    """Default scale and structure for the ArthJAX synthetic economy."""

    num_households: int = 250
    num_companies: int = 60
    num_sectors: int = 10
    num_commodities: int = 6
    num_currencies: int = 4
    num_agent_types: int = 4  # value, momentum, panic, saver
    contagion_steps: int = 3

    # Simulation defaults
    default_num_steps: int = 600
    default_seed: int = 42

    # Macro anchors
    initial_interest_rate: float = 0.04
    initial_inflation: float = 0.02
    initial_unemployment: float = 0.05
    initial_sentiment: float = 0.5
    nairu: float = 0.05

    # World model defaults (v0.3)
    world_model_hidden_dim: int = 128
    world_model_epochs: int = 50
    world_model_batch_size: int = 64
    world_model_lr: float = 0.002


DEFAULT_CONFIG = EconomyConfig()
