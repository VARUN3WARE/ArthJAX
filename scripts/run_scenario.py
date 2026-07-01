#!/usr/bin/env python3
"""Run a named ArthJAX stress scenario."""

from __future__ import annotations

import argparse
import time

import jax
import numpy as np

from arthjax import EconomyConfig, __version__, init_state, simulate
from arthjax.scenarios import SCENARIO_NAMES, apply_scenario_to_state, resolve_scenario
from arthjax.scenarios.presets import _PRESETS
from arthjax.simulation.step import make_step_jit
from arthjax.viz.output import DEFAULT_PLOTS_DIR, resolve_output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a named ArthJAX scenario")
    parser.add_argument(
        "--scenario",
        type=str,
        default="baseline",
        choices=list(SCENARIO_NAMES),
        help="Scenario preset name",
    )
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--list", action= "store_true")
    args = parser.parse_args()

    if args.list:
        print("Available scenarios and descriptions:")
        for scenario in SCENARIO_NAMES:
            print(f"  {scenario}")
            print(f"        {_PRESETS[scenario].description}\n")
        return 0

    base = EconomyConfig(default_num_steps=args.steps, default_seed=args.seed)
    cfg, preset = resolve_scenario(args.scenario, base)

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)

    print(f"ArthJAX v{__version__} — scenario: {preset.name}")
    print(f"  {preset.description}")

    state = init_state(cfg, init_key)
    state = apply_scenario_to_state(state, preset, cfg)
    step_jit = make_step_jit(cfg)

    start = time.time()
    _, metrics_history = simulate(state, args.steps, key, cfg=cfg, step_jit=step_jit)
    elapsed = time.time() - start

    metrics_np = jax.tree.map(np.array, metrics_history)
    print(f"Done in {elapsed:.2f}s")
    print(f"GDP: ${metrics_np['gdp'][-1]:.0f}")
    print(f"Credit stress: {metrics_np['credit_stress'][-1]*100:.2f}%")
    print(f"Bad loan ratio: {metrics_np['bad_loan_ratio'][-1]*100:.2f}%")
    print(f"Sentiment: {metrics_np['sentiment'][-1]:.3f}")

    if args.plot:
        from arthjax.viz.plots import save_all_plots

        out = resolve_output_dir(args.output_dir)
        save_all_plots(metrics_np, out, cfg)
        print(f"Plots saved to {out}")

    return 0






if __name__ == "__main__":
    raise SystemExit(main())
