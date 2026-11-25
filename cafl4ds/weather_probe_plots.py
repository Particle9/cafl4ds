"""Shared plot order and styling for the weather-probe figures."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes

PLOT_ARMS = (
    "vmoe_top4",
    "vmoe_top1",
    "gmm_capped",
    "gmm_all_layers",
    "gmm_split",
    "gmm_full",
    "gmm_full_private_decoder",
)
WITHIN = "#14645F"
CROSS = "#AD4825"
INK = "#26394A"
BASELINE = "#676C79"
GRID = "#E1E7EB"


def style_probe_axis(ax: Axes, labels: list[str]) -> None:
    """Format the architecture axis consistently across weather-probe figures."""
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.48, -0.52)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0, pad=8)
