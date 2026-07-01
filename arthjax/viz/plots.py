"""Macro and world-model visualization."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from arthjax.config import EconomyConfig, DEFAULT_CONFIG
from arthjax.viz.style import PALETTE, apply_plot_style, rolling_std, style_ax


def plot_macro_evolution(metrics_np: dict, output_path: str, burn: int = 25) -> None:
    apply_plot_style()
    T = np.arange(burn, len(metrics_np["gdp"]))

    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(
        "ArthJAX · Synthetic Economy Dynamics",
        fontsize=18,
        fontweight="bold",
        color="#f0f6fc",
        y=0.98,
    )

    plots = [
        (axes[0, 0], metrics_np["gdp"][burn:], "GDP", "Output ($)", PALETTE["gdp"]),
        (
            axes[0, 1],
            metrics_np["inflation"][burn:] * 100,
            "Inflation",
            "%",
            PALETTE["inflation"],
        ),
        (
            axes[0, 2],
            metrics_np["unemployment"][burn:] * 100,
            "Unemployment",
            "%",
            PALETTE["unemployment"],
        ),
        (
            axes[1, 0],
            metrics_np["interest_rate"][burn:] * 100,
            "Interest Rate",
            "%",
            PALETTE["rates"],
        ),
        (
            axes[1, 1],
            metrics_np["stock_index"][burn:],
            "Stock Index",
            "Index",
            PALETTE["stocks"],
        ),
        (
            axes[1, 2],
            metrics_np["gini_coefficient"][burn:],
            "Wealth Inequality",
            "Gini",
            PALETTE["gini"],
        ),
        (
            axes[2, 0],
            metrics_np["sentiment"][burn:],
            "Market Sentiment",
            "Index",
            PALETTE["sentiment"],
        ),
        (
            axes[2, 1],
            metrics_np["credit_stress"][burn:] * 100,
            "Credit Stress",
            "%",
            PALETTE["stress"],
        ),
        (
            axes[2, 2],
            metrics_np["bad_loan_ratio"][burn:] * 100,
            "Bad Loan Ratio",
            "%",
            PALETTE["loans"],
        ),
    ]

    for ax, data, title, ylab, color in plots:
        ax.plot(T, data, linewidth=2.2, color=color)
        ax.fill_between(T, data, alpha=0.12, color=color)
        style_ax(ax, title, ylab)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="#0f1117")
    plt.close()


def plot_boom_bust(metrics_np: dict, output_path: str, burn: int = 25) -> None:
    apply_plot_style()
    T = np.arange(burn, len(metrics_np["gdp"]))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(
        "Emergent Boom-Bust Cycles & Macro Contagion",
        fontsize=16,
        fontweight="bold",
        color="#f0f6fc",
        y=0.98,
    )

    ax = axes[0, 0]
    ax.set_facecolor("#161b22")
    ax.plot(T, metrics_np["gdp"][burn:], linewidth=2.5, color=PALETTE["gdp"], label="GDP")
    ax2 = ax.twinx()
    ax2.plot(
        T,
        metrics_np["credit_stress"][burn:] * 100,
        linewidth=2,
        color=PALETTE["stress"],
        linestyle="--",
        alpha=0.85,
        label="Credit Stress",
    )
    ax.set_title("GDP vs Credit Stress", fontweight="bold", color="#f0f6fc")
    ax.set_ylabel("GDP ($)", color=PALETTE["gdp"])
    ax2.set_ylabel("Credit Stress (%)", color=PALETTE["stress"])
    ax2.tick_params(colors="#8b949e")
    ax.grid(True, alpha=0.35, color="#21262d")
    ax.tick_params(colors="#8b949e")

    ax = axes[0, 1]
    ax.set_facecolor("#161b22")
    s = metrics_np["sentiment"][burn:]
    ax.plot(T, s, linewidth=2.5, color=PALETTE["sentiment"])
    ax.fill_between(T, s, 0.5, where=(s > 0.5), alpha=0.15, color="#3fb950")
    ax.fill_between(T, s, 0.5, where=(s <= 0.5), alpha=0.15, color="#f85149")
    ax.axhline(0.5, color="#484f58", linestyle=":", alpha=0.7)
    ax.set_title("Sentiment Cycles", fontweight="bold", color="#f0f6fc")
    ax.set_ylabel("Sentiment Index")
    ax.set_ylim(0.25, 0.75)
    ax.grid(True, alpha=0.35, color="#21262d")
    ax.tick_params(colors="#8b949e")

    ax = axes[1, 0]
    ax.set_facecolor("#161b22")
    sc = ax.scatter(
        metrics_np["unemployment"][burn:] * 100,
        metrics_np["inflation"][burn:] * 100,
        c=T,
        cmap="plasma",
        s=28,
        alpha=0.75,
        edgecolors="none",
    )
    ax.set_title("Phillips Curve", fontweight="bold", color="#f0f6fc")
    ax.set_xlabel("Unemployment (%)", color="#8b949e")
    ax.set_ylabel("Inflation (%)", color="#8b949e")
    ax.grid(True, alpha=0.35, color="#21262d")
    ax.tick_params(colors="#8b949e")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Time Step", color="#8b949e")
    cb.ax.yaxis.set_tick_params(color="#8b949e")

    ax = axes[1, 1]
    ax.set_facecolor("#161b22")
    rw = 40
    gv = rolling_std(metrics_np["gdp"][burn:], rw)
    rv = rolling_std(metrics_np["interest_rate"][burn:], rw)
    T2 = T[1:]
    ax.plot(T2, gv, linewidth=2, color=PALETTE["gdp"], label=f"GDP vol ({rw}-step)")
    ax.plot(
        T2, rv * 100, linewidth=2, color=PALETTE["rates"], label=f"Rate vol ({rw}-step)"
    )
    ax.set_title("Macro Volatility", fontweight="bold", color="#f0f6fc")
    ax.set_ylabel("Volatility")
    ax.legend(
        fontsize=8, facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9"
    )
    ax.grid(True, alpha=0.35, color="#21262d")
    ax.tick_params(colors="#8b949e")

    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="#0f1117")
    plt.close()


def plot_linkedin_hero(metrics_np: dict, output_path: str, burn: int = 25) -> None:
    apply_plot_style()
    T = np.arange(burn, len(metrics_np["gdp"]))

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#161b22")

    ax2 = ax.twinx()
    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("outward", 55))

    l1, = ax.plot(T, metrics_np["gdp"][burn:], color=PALETTE["gdp"], linewidth=2.8, label="GDP ($)")
    l2, = ax2.plot(
        T,
        metrics_np["stock_index"][burn:],
        color=PALETTE["stocks"],
        linewidth=2.2,
        label="Stock Index",
        alpha=0.9,
    )
    l3, = ax3.plot(
        T,
        metrics_np["unemployment"][burn:] * 100,
        color=PALETTE["unemployment"],
        linewidth=2.2,
        linestyle="--",
        label="Unemployment (%)",
        alpha=0.9,
    )

    ax.set_xlabel("Time Step", color="#8b949e", fontsize=11)
    ax.set_ylabel("GDP ($)", color=PALETTE["gdp"], fontsize=11)
    ax2.set_ylabel("Stock Index", color=PALETTE["stocks"], fontsize=11)
    ax3.set_ylabel("Unemployment (%)", color=PALETTE["unemployment"], fontsize=11)
    ax.tick_params(colors="#8b949e")
    ax2.tick_params(colors="#8b949e")
    ax3.tick_params(colors="#8b949e")

    for spine in ax.spines.values():
        spine.set_color("#30363d")
    for spine in ax2.spines.values():
        spine.set_color("#30363d")
    ax3.spines["right"].set_color("#30363d")

    ax.set_title(
        "ArthJAX · Emergent Macro Dynamics from 250 AI Agents",
        fontsize=16,
        fontweight="bold",
        color="#f0f6fc",
        pad=14,
    )
    ax.grid(True, alpha=0.3, color="#21262d")

    stress = metrics_np["credit_stress"][burn:]
    if np.max(stress) > 0.15:
        idx = np.argmax(stress)
        ax.axvline(T[idx], color="#f85149", alpha=0.35, linestyle=":", linewidth=1.5)

    fig.legend(
        [l1, l2, l3],
        ["GDP ($)", "Stock Index", "Unemployment (%)"],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.02),
        facecolor="#161b22",
        edgecolor="#30363d",
        labelcolor="#c9d1d9",
        fontsize=10,
    )

    plt.tight_layout(rect=[0.04, 0.06, 1, 1])
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#0f1117")
    plt.close()


def plot_world_model_loss(losses: list[float], output_path: str) -> None:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#161b22")
    ax.plot(losses, linewidth=2.5, color=PALETTE["gdp"])
    ax.set_xlabel("Epoch", fontweight="bold", color="#8b949e")
    ax.set_ylabel("MSE Loss (normalized)", fontweight="bold", color="#8b949e")
    ax.set_title("World Model Training Convergence", fontweight="bold", color="#f0f6fc")
    ax.grid(True, alpha=0.35, color="#21262d")
    ax.set_yscale("log")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="#0f1117")
    plt.close()


def _smooth(y: np.ndarray, w: int = 5) -> np.ndarray:
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="same")


def plot_real_vs_learned(
    real_trajectory: np.ndarray,
    learned_trajectory: np.ndarray,
    output_path: str,
) -> None:
    apply_plot_style()
    idx = real_trajectory.shape[1] - 4
    ts = np.arange(len(real_trajectory))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(
        "Neural World Model · Real vs Learned Economy",
        fontsize=16,
        fontweight="bold",
        color="#f0f6fc",
        y=0.98,
    )

    macro_labels = [
        ("Interest Rate", "%", 0),
        ("Inflation", "%", 1),
        ("Unemployment", "%", 2),
        ("Market Sentiment", "index", 3),
    ]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for (title, ylab, off), (row, col) in zip(macro_labels, positions):
        ax = axes[row, col]
        ax.set_facecolor("#161b22")
        rv = real_trajectory[:, idx + off]
        lv = learned_trajectory[:, idx + off]
        if ylab == "%":
            rv, lv = rv * 100, lv * 100
        ax.plot(ts, rv, label="Simulator", linewidth=2.5, color=PALETTE["real"], alpha=0.95)
        ax.plot(
            ts,
            _smooth(lv),
            label="World Model",
            linewidth=2,
            color=PALETTE["learned"],
            linestyle="--",
            alpha=0.9,
        )
        ax.set_title(title, fontweight="bold", color="#f0f6fc")
        ax.set_ylabel(ylab, color="#8b949e")
        ax.set_xlabel("Time Step", color="#8b949e")
        ax.legend(
            facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9
        )
        ax.grid(True, alpha=0.35, color="#21262d")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="#0f1117")
    plt.close()


def plot_policy_counterfactual(
    metrics_a: dict,
    metrics_b: dict,
    label_a: str,
    label_b: str,
    output_path: str,
    burn: int = 25,
) -> None:
    """Overlay macro paths for two policies on the same seed."""
    apply_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(
        "Policy counterfactual (same seed)",
        fontsize=16,
        fontweight="bold",
        color="#f0f6fc",
        y=0.98,
    )

    series = [
        (axes[0, 0], "gdp", "GDP ($)", PALETTE["gdp"]),
        (axes[0, 1], "inflation", "Inflation (%)", PALETTE["inflation"]),
        (axes[1, 0], "unemployment", "Unemployment (%)", PALETTE["unemployment"]),
        (axes[1, 1], "interest_rate", "Interest rate (%)", PALETTE["rates"]),
    ]
    for ax, key, ylab, color_a in series:
        a = metrics_a[key][burn:]
        b = metrics_b[key][burn:]
        scale = 100.0 if key != "gdp" else 1.0
        t = np.arange(burn, burn + len(a))
        ax.plot(t, a * scale, color=color_a, linewidth=1.8, label=label_a)
        ax.plot(t, b * scale, color=PALETTE["learned"], linewidth=1.8, linestyle="--", label=label_b)
        style_ax(ax, ylab.replace(" (%)", "").replace(" ($)", ""), ylab, xlabel="Step")
        ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#f0f6fc", fontsize=8)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()


def save_all_plots(
    metrics_np: dict,
    output_dir: str,
    cfg: EconomyConfig = DEFAULT_CONFIG,
) -> None:
    burn = cfg.plot_burn_in
    plot_macro_evolution(
        metrics_np, os.path.join(output_dir, "macro_evolution_v2.png"), burn=burn
    )
    plot_boom_bust(metrics_np, os.path.join(output_dir, "boom_bust_v2.png"), burn=burn)
    plot_linkedin_hero(metrics_np, os.path.join(output_dir, "linkedin_hero.png"), burn=burn)
