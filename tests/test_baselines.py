"""Baseline comparison table tests."""

import numpy as np

from arthjax.benchmarks.baselines import build_comparison_table, format_comparison_table


def test_comparison_table_includes_all_methods():
    real = np.random.randn(60, 4) * 0.05 + np.array([0.04, 0.02, 0.05, 0.5])
    rows = build_comparison_table(
        real_macro=real,
        full_sim_sec=1.5,
        sim_steps=600,
        wm_mae=0.02,
        wm_relative_pct=15.0,
        wm_rollout_sec=0.1,
        horizon=30,
    )
    methods = {r.method for r in rows}
    assert "Full simulation (reference)" in methods
    assert "AR(1) rolled forward" in methods
    assert "World model (macro)" in methods
    text = format_comparison_table(rows)
    assert "Forecast comparison" in text
    assert "World model (macro)" in text
