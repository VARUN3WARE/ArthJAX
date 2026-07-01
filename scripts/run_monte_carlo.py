#!/usr/bin/env python3
"""Monte Carlo stress paths over multiple seeds."""

from __future__ import annotations

import argparse
import time

import jax
import jax.random as jr
import numpy as np

from arthjax import EconomyConfig, __version__, init_state, simulate
from arthjax.scenarios import apply_scenario_to_state, resolve_scenario
from arthjax.simulation.step import make_step_jit


def main() -> int:
    parser = argparse.ArgumentParser(description="ArthJAX Monte Carlo stress runs")
    parser.add_argument("--scenario", type=str, default="credit_crunch")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Tiny economy for smoke")
    args = parser.parse_args()

    if args.quick:
        base = EconomyConfig(
            num_households=20,
            num_companies=10,
            num_sectors=5,
            default_num_steps=args.steps,
        )
    else:
        base = EconomyConfig(default_num_steps=args.steps)

    cfg, preset = resolve_scenario(args.scenario, base)
    step_jit = make_step_jit(cfg)

    print(f"ArthJAX v{__version__} — Monte Carlo ({args.runs} runs, {args.scenario})")
    t0 = time.time()

    final_gdp, final_unemp, final_credit, final_rates = [], [], [], []
    key = jr.PRNGKey(args.seed)

    for i in range(args.runs):
        key, init_key, run_key = jr.split(key, 3)
        state = init_state(cfg, init_key)
        state = apply_scenario_to_state(state, preset, cfg)
        _, metrics = simulate(state, args.steps, run_key, cfg=cfg, step_jit=step_jit)
        m = jax.tree.map(np.array, metrics)
        final_gdp.append(float(m["gdp"][-1]))
        final_unemp.append(float(m["unemployment"][-1]))
        final_credit.append(float(m["credit_stress"][-1]))
        final_rates.append(float(m["interest_rate"][-1]))

    elapsed = time.time() - t0
    gdp = np.array(final_gdp)
    unemp = np.array(final_unemp)
    credit = np.array(final_credit)
    rates = np.array(final_rates)

    print(f"Completed in {elapsed:.2f}s ({args.runs / max(elapsed, 1e-6):.1f} runs/s)")
    print(f"GDP final:     mean={gdp.mean():.0f}  std={gdp.std():.0f}")
    print(f"Unemployment:  mean={unemp.mean()*100:.2f}%  std={unemp.std()*100:.2f}%")
    print(f"Credit stress: mean={credit.mean()*100:.2f}%  std={credit.std()*100:.2f}%")
    print(f"Interest rate: mean={rates.mean()*100:.2f}%  std={rates.std()*100:.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
