#!/usr/bin/env python3
"""Train neural world model on simulation rollouts."""

from __future__ import annotations

import argparse
import os
import time

import jax
import jax.random as jr
import numpy as np

from arthjax import EconomyConfig, __version__, init_state
from arthjax.simulation.step import make_step_jit
from arthjax.world_model.rollout import (
    collect_real_macro_trajectory,
    collect_real_trajectory,
    compare_macro_rollouts,
    compare_rollouts,
    rollout_learned,
)
from arthjax.world_model.train import train_world_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ArthJAX neural world model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--rollouts", type=int, default=None)
    parser.add_argument("--rollout-length", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    cfg_kwargs = {}
    if args.epochs is not None:
        cfg_kwargs["world_model_epochs"] = args.epochs
    if args.rollouts is not None:
        cfg_kwargs["world_model_num_rollouts"] = args.rollouts
    if args.rollout_length is not None:
        cfg_kwargs["world_model_rollout_length"] = args.rollout_length

    cfg = EconomyConfig(**cfg_kwargs)
    key = jr.PRNGKey(args.seed)
    key, init_key = jr.split(key)

    print(f"ArthJAX v{__version__} — world model training")
    state = init_state(cfg, init_key)
    step_jit = make_step_jit(cfg)

    start = time.time()
    params, normalizer, losses = train_world_model(
        state, step_jit, cfg, key, macro_only=cfg.world_model_macro_only
    )
    print(f"Training done in {time.time() - start:.1f}s, final loss={losses[-1]:.6f}")

    key, eval_key = jr.split(key)
    if cfg.world_model_macro_only:
        real_traj = collect_real_macro_trajectory(
            state, step_jit, cfg, jr.PRNGKey(cfg.world_model_eval_seed)
        )
        learned_traj = rollout_learned(
            params,
            real_traj[0],
            normalizer,
            cfg,
            num_steps=len(real_traj) - 1,
            macro_only=True,
        )
        metrics = compare_macro_rollouts(real_traj, learned_traj)
        print(
            f"Macro-only MAE: {metrics['macro_mae']:.6f} "
            f"({metrics['macro_relative_error_pct']:.2f}% relative)"
        )
    else:
        real_traj = collect_real_trajectory(state, step_jit, cfg, eval_key)
        learned_traj = rollout_learned(
            params, real_traj[0], normalizer, cfg, num_steps=len(real_traj) - 1
        )
        metrics = compare_rollouts(real_traj, learned_traj)
        print(f"Full-state MAE: {metrics['mae']:.6f} ({metrics['relative_error_pct']:.2f}% relative)")
        print(
            f"Macro-only MAE: {metrics['macro_mae']:.6f} "
            f"({metrics['macro_relative_error_pct']:.2f}% relative)"
        )

    if args.plot:
        from arthjax.viz.plots import plot_real_vs_learned, plot_world_model_loss

        out = os.path.abspath(args.output_dir)
        os.makedirs(out, exist_ok=True)
        plot_world_model_loss(losses, os.path.join(out, "world_model_loss_v2.png"))
        plot_real_vs_learned(
            real_traj, learned_traj, os.path.join(out, "real_vs_learned_v2.png")
        )
        print(f"Plots saved to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
