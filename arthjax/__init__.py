"""ArthJAX — GPU-accelerated synthetic macro economy in JAX."""

__version__ = "1.1.0"
__all__ = [
    "__version__",
    "EconomyConfig",
    "init_state",
    "STATE_KEYS",
    "METRIC_KEYS",
    "simulate",
]

from arthjax.config import EconomyConfig
from arthjax.simulation.loop import simulate
from arthjax.state import METRIC_KEYS, STATE_KEYS, init_state
