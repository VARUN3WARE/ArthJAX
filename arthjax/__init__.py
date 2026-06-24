"""ArthJAX — GPU-accelerated synthetic macro economy in JAX."""

__version__ = "0.1.0"
__all__ = ["__version__", "EconomyConfig", "init_state", "STATE_KEYS", "METRIC_KEYS"]

from arthjax.config import EconomyConfig
from arthjax.state import METRIC_KEYS, STATE_KEYS, init_state
