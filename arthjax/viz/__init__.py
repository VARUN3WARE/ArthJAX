"""Visualization subpackage."""

from arthjax.viz.plots import (
    plot_boom_bust,
    plot_linkedin_hero,
    plot_macro_evolution,
    plot_real_vs_learned,
    plot_world_model_loss,
    save_all_plots,
)
from arthjax.viz.style import PALETTE, PLOT_STYLE, apply_plot_style, rolling_std, style_ax

__all__ = [
    "PALETTE",
    "PLOT_STYLE",
    "apply_plot_style",
    "rolling_std",
    "style_ax",
    "plot_macro_evolution",
    "plot_boom_bust",
    "plot_linkedin_hero",
    "plot_world_model_loss",
    "plot_real_vs_learned",
    "save_all_plots",
]
