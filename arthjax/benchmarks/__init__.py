"""Benchmarks subpackage."""

from arthjax.benchmarks.baselines import BaselineResult, evaluate_ar1_baseline
from arthjax.benchmarks.stylized_facts import (
    StylizedFactResult,
    compute_stylized_facts,
    summarize_facts,
)

__all__ = [
    "StylizedFactResult",
    "compute_stylized_facts",
    "summarize_facts",
    "BaselineResult",
    "evaluate_ar1_baseline",
]
