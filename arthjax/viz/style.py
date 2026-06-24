"""Plot styling for ArthJAX charts."""

import matplotlib.pyplot as plt
import numpy as np

PLOT_STYLE = {
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "font.family": "sans-serif",
    "font.size": 10,
}

PALETTE = {
    "gdp": "#58a6ff",
    "inflation": "#f778ba",
    "unemployment": "#ffa657",
    "rates": "#ff7b72",
    "stocks": "#3fb950",
    "gini": "#d2a8ff",
    "sentiment": "#79c0ff",
    "stress": "#f85149",
    "loans": "#e3b341",
    "real": "#58a6ff",
    "learned": "#ffa657",
}


def apply_plot_style() -> None:
    plt.rcParams.update(PLOT_STYLE)


def style_ax(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor("#161b22")
    ax.set_title(title, fontweight="bold", fontsize=11, color="#f0f6fc", pad=8)
    ax.set_ylabel(ylabel, fontsize=9, color="#8b949e")
    ax.set_xlabel("Time Step", fontsize=9, color="#8b949e")
    ax.grid(True, alpha=0.35, color="#21262d")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")


def rolling_std(series: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros(len(series) - 1)
    for i in range(1, len(series)):
        seg = series[max(0, i - window) : i]
        out[i - 1] = float(np.std(seg)) if len(seg) > 1 else 0.0
    return out
