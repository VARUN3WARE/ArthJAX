#!/usr/bin/env python3
"""Run the ArthJAX synthetic economy simulation."""

from __future__ import annotations

import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from arthjax import EconomyConfig, __version__, init_state
from arthjax.simulation.loop import simulate
from arthjax.simulation.step import make_step_jit


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ArthJAX macro ABM simulation")
    parser.add_argument("--steps", type=int, default=600, help="Number of timesteps")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for plot output",
    )
    parser.add_argument("--plot", action="store_true", help="Save macro charts")
    args = parser.parse_args()

    cfg = EconomyConfig(
        default_num_steps=args.steps,
        default_seed=args.seed,
    )
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)

    print(f"ArthJAX v{__version__}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Running {args.steps} steps...")

    state = init_state(cfg, init_key)
    step_jit = make_step_jit(cfg)

    start = time.time()
    final_state, metrics_history = simulate(
        state, args.steps, key, cfg=cfg, step_jit=step_jit
    )
    elapsed = time.time() - start

    metrics_np = jax.tree.map(np.array, metrics_history)
    print(f"Done in {elapsed:.2f}s ({args.steps / elapsed:.1f} steps/sec)")
    print(f"GDP: ${metrics_np['gdp'][-1]:.0f}")
    print(f"Inflation: {metrics_np['inflation'][-1] * 100:.2f}%")
    print(f"Unemployment: {metrics_np['unemployment'][-1] * 100:.2f}%")
    print(f"Sentiment: {metrics_np['sentiment'][-1]:.3f}")
    print(f"Gini: {metrics_np['gini_coefficient'][-1]:.4f}")

    if args.plot:
        from arthjax.viz.plots import save_all_plots

        out = os.path.abspath(args.output_dir)
        os.makedirs(out, exist_ok=True)
        save_all_plots(metrics_np, out, cfg)
        print(f"Plots saved to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
