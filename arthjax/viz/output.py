"""Default plot output directory."""

from __future__ import annotations

import os

DEFAULT_PLOTS_DIR = "plots"


def resolve_output_dir(path: str | None = None) -> str:
    """Return absolute output directory, creating it if needed."""
    out = os.path.abspath(path or DEFAULT_PLOTS_DIR)
    os.makedirs(out, exist_ok=True)
    return out
