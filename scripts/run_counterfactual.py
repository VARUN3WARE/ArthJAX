#!/usr/bin/env python3
"""Run same-seed policy counterfactual comparison."""

from __future__ import annotations

import argparse
import os
import time

import jax
import jax.random as jr
import numpy as np

from arthjax import EconomyConfig, __version__, init_state, simulate
from arthjax.policy import POLICY_NAMES, resolve_policy
from arthjax.scenarios import apply_scenario_to_state, resolve_scenario
from arthjax.simulation.step import make_step_jit
from arthjax.viz.output import DEFAULT_PLOTS_DIR, resolve_output_dir
from arthjax.viz.plots import plot_policy_counterfactual


def _run_policy(
    policy_name: str,
    scenario_name: str,
    steps: int,
    seed: int,
    init_key,
):
    base = EconomyConfig(default_num_steps=steps, default_seed=seed)
    cfg_scenario, preset = resolve_scenario(scenario_name, base)
    cfg, policy = resolve_policy(policy_name, cfg_scenario)
    state = init_state(cfg, init_key)
    state = apply_scenario_to_state(state, preset, cfg)
    step_jit = make_step_jit(cfg)
    key = jr.PRNGKey(seed)
    _, metrics = simulate(state, steps, key, cfg=cfg, step_jit=step_jit)
    return policy, jax.tree.map(np.array, metrics)


def main() -> int:
    parser = argparse.ArgumentParser(description="ArthJAX policy counterfactual")
    parser.add_argument("--policy-a", type=str, default="baseline", choices=list(POLICY_NAMES))
    parser.add_argument("--policy-b", type=str, default="hawkish", choices=list(POLICY_NAMES))
    parser.add_argument("--scenario", type=str, default="baseline")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    print(f"ArthJAX v{__version__} — policy counterfactual")
    print(f"  scenario={args.scenario}  seed={args.seed}")
    print(f"  A: {args.policy_a}  vs  B: {args.policy_b}")

    key = jr.PRNGKey(args.seed)
    _, init_key = jr.split(key)

    t0 = time.time()
    pol_a, metrics_a = _run_policy(
        args.policy_a, args.scenario, args.steps, args.seed, init_key
    )
    pol_b, metrics_b = _run_policy(
        args.policy_b, args.scenario, args.steps, args.seed, init_key
    )
    elapsed = time.time() - t0

    print(f"Done in {elapsed:.2f}s")
    print(
        f"Final unemployment A={metrics_a['unemployment'][-1]*100:.2f}% "
        f"B={metrics_b['unemployment'][-1]*100:.2f}%"
    )
    print(
        f"Final interest rate A={metrics_a['interest_rate'][-1]*100:.2f}% "
        f"B={metrics_b['interest_rate'][-1]*100:.2f}%"
    )

    if args.plot:
        out = resolve_output_dir(args.output_dir)
        path = os.path.join(out, "policy_counterfactual.png")
        plot_policy_counterfactual(
            metrics_a,
            metrics_b,
            pol_a.name,
            pol_b.name,
            path,
            burn=EconomyConfig().plot_burn_in,
        )
        print(f"Plot saved to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
