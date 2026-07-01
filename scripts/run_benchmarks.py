#!/usr/bin/env python3
"""Run stylized-facts benchmarks on a default simulation."""

from __future__ import annotations

import argparse
import os
import time

import jax
import jax.random as jr
import numpy as np

from arthjax import EconomyConfig, __version__, init_state, simulate
from arthjax.benchmarks import (
    build_comparison_table,
    compute_stylized_facts,
    format_comparison_table,
    summarize_facts,
)
from arthjax.benchmarks.baselines import compare_speed
from arthjax.benchmarks.plots import plot_phillips_scatter, plot_volatility_clustering
from arthjax.simulation.step import make_step_jit
from arthjax.viz.output import DEFAULT_PLOTS_DIR, resolve_output_dir
from arthjax.world_model.rollout import (
    collect_real_macro_trajectory,
    compare_macro_rollouts,
    rollout_learned,
)
from arthjax.world_model.train import train_world_model


def main() -> int:
    parser = argparse.ArgumentParser(description="ArthJAX stylized-facts benchmarks")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Tiny economy for CI smoke")
    parser.add_argument("--plot", action="store_true", help="Save Phillips + vol clustering charts")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_PLOTS_DIR)
    args = parser.parse_args()

    if args.quick:
        cfg = EconomyConfig(
            num_households=20,
            num_companies=10,
            num_sectors=5,
            default_num_steps=args.steps,
            world_model_epochs=3,
            world_model_num_rollouts=2,
            world_model_rollout_length=30,
        )
    else:
        cfg = EconomyConfig(default_num_steps=args.steps, default_seed=args.seed)

    print(f"ArthJAX v{__version__} — benchmarks")
    key = jr.PRNGKey(args.seed)
    key, init_key = jr.split(key)
    state = init_state(cfg, init_key)
    step_jit = make_step_jit(cfg)

    t0 = time.time()
    _, metrics_history = simulate(state, args.steps, key, cfg=cfg, step_jit=step_jit)
    sim_sec = time.time() - t0
    metrics_np = jax.tree.map(np.array, metrics_history)

    facts = compute_stylized_facts(metrics_np, burn=cfg.plot_burn_in)
    print(summarize_facts(facts))
    print()

    if args.plot:
        out = resolve_output_dir(args.output_dir)
        plot_phillips_scatter(
            metrics_np, os.path.join(out, "phillips_scatter.png"), burn=cfg.plot_burn_in
        )
        plot_volatility_clustering(
            metrics_np, os.path.join(out, "vol_clustering.png"), burn=cfg.plot_burn_in
        )
        print(f"Plots saved to {out}")

    horizon = min(50, 99)
    real_macro = collect_real_macro_trajectory(
        state, step_jit, cfg, jr.PRNGKey(cfg.world_model_eval_seed), num_steps=horizon + 1
    )

    wm_key = jr.PRNGKey(100)
    t1 = time.time()
    params, normalizer, _ = train_world_model(state, step_jit, cfg, wm_key, macro_only=True)
    train_sec = time.time() - t1

    t2 = time.time()
    learned = rollout_learned(
        params,
        real_macro[0],
        normalizer,
        cfg,
        num_steps=len(real_macro) - 1,
        macro_only=True,
    )
    wm_sec = time.time() - t2
    wm_metrics = compare_macro_rollouts(real_macro, learned)
    print(
        f"World model macro MAE: {wm_metrics['macro_mae']:.6f} "
        f"({wm_metrics['macro_relative_error_pct']:.2f}% relative)"
    )

    table = build_comparison_table(
        real_macro,
        full_sim_sec=sim_sec,
        sim_steps=args.steps,
        wm_mae=wm_metrics["macro_mae"],
        wm_relative_pct=wm_metrics["macro_relative_error_pct"],
        wm_rollout_sec=wm_sec,
        horizon=min(50, len(real_macro) - 1),
    )
    print()
    print(format_comparison_table(table))
    print()
    print(compare_speed(sim_sec, wm_sec, args.steps))
    print(f"WM train time: {train_sec:.2f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
