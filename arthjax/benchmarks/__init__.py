"""Benchmarks subpackage."""

from arthjax.benchmarks.baselines import (
    BaselineResult,
    ComparisonRow,
    build_comparison_table,
    evaluate_ar1_baseline,
    format_comparison_table,
)
from arthjax.benchmarks.plots import plot_phillips_scatter, plot_volatility_clustering
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
    "ComparisonRow",
    "build_comparison_table",
    "evaluate_ar1_baseline",
    "format_comparison_table",
    "plot_phillips_scatter",
    "plot_volatility_clustering",
]
