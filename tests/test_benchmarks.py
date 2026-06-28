"""Benchmark and stylized-facts tests."""

import jax
import jax.random as jr
import numpy as np
import pytest

from arthjax import EconomyConfig, init_state, simulate
from arthjax.benchmarks import compute_stylized_facts
from arthjax.benchmarks.stylized_facts import (
    rolling_return_volatility,
    volatility_clustering_score,
)
from arthjax.simulation.step import make_step_jit


@pytest.fixture
def default_metrics():
    cfg = EconomyConfig(default_num_steps=200, default_seed=42)
    key = jr.PRNGKey(cfg.default_seed)
    state = init_state(cfg, jr.split(key)[0])
    _, mh = simulate(state, 200, key, cfg=cfg, step_jit=make_step_jit(cfg))
    return jax.tree.map(np.array, mh), cfg


def test_volatility_clustering_on_default_run(default_metrics):
    metrics_np, cfg = default_metrics
    gdp_score, ret_autocorr = volatility_clustering_score(metrics_np, burn=cfg.plot_burn_in)
    assert gdp_score > 5.0
    assert ret_autocorr > 0.2


def test_rolling_return_volatility_finite(default_metrics):
    metrics_np, cfg = default_metrics
    rv = rolling_return_volatility(metrics_np["stock_index"], burn=cfg.plot_burn_in)
    assert len(rv) > 10
    assert np.all(np.isfinite(rv))
    assert np.std(rv) > 1e-6


def test_stylized_facts_volatility_autocorr_passes(default_metrics):
    metrics_np, cfg = default_metrics
    facts = compute_stylized_facts(metrics_np, burn=cfg.plot_burn_in)
    assert "volatility_autocorr" in facts
    assert facts["volatility_autocorr"].passed


def test_benchmark_plots_smoke(default_metrics, tmp_path):
    from arthjax.benchmarks.plots import plot_phillips_scatter, plot_volatility_clustering

    metrics_np, cfg = default_metrics
    plot_phillips_scatter(metrics_np, str(tmp_path / "phillips.png"), burn=cfg.plot_burn_in)
    plot_volatility_clustering(metrics_np, str(tmp_path / "vol.png"), burn=cfg.plot_burn_in)
    assert (tmp_path / "phillips.png").exists()
    assert (tmp_path / "vol.png").exists()
