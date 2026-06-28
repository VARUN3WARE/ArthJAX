"""Benchmark visualization helpers."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from arthjax.viz.style import PALETTE, apply_plot_style, style_ax


def plot_phillips_scatter(
    metrics_np: dict,
    output_path: str,
    burn: int = 25,
) -> None:
    """Scatter inflation vs unemployment with OLS fit line."""
    apply_plot_style()
    unemp = metrics_np["unemployment"][burn:] * 100
    infl = metrics_np["inflation"][burn:] * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.scatter(unemp, infl, alpha=0.55, s=18, c=PALETTE["inflation"], edgecolors="none")
    style_ax(ax, "Phillips curve (synthetic)", "Inflation (%)", "Unemployment (%)")

    if len(unemp) > 10 and np.std(unemp) > 1e-6:
        slope, intercept = np.polyfit(unemp, infl, 1)
        x_line = np.linspace(unemp.min(), unemp.max(), 50)
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color=PALETTE["rates"],
            linewidth=2,
            label=f"slope={slope:.3f}",
        )
        ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#f0f6fc")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_volatility_clustering(
    metrics_np: dict,
    output_path: str,
    burn: int = 25,
) -> None:
    """Rolling return volatility series used for the autocorr stylized fact."""
    apply_plot_style()
    si = metrics_np["stock_index"][burn:]
    rolling_ret_vol = []
    for i in range(40, len(si)):
        window = si[i - 40 : i + 1]
        rolling_ret_vol.append(float(np.std(np.diff(np.log(window + 1e-8)))))
    rolling_ret_vol = np.array(rolling_ret_vol)
    t = np.arange(len(rolling_ret_vol)) + burn + 40

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0f1117")
    ax.plot(t, rolling_ret_vol, color=PALETTE["stocks"], linewidth=1.5)
    style_ax(ax, "Volatility clustering (rolling 40-step return vol)", "Return vol", "Step")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
