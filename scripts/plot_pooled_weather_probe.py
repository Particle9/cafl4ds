"""Plot class-balanced linear probes from the pooled M6.4.1 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from cafl4ds.day_night_analysis import LABELS
from cafl4ds.weather_probe_plots import INK, PLOT_ARMS, WITHIN, style_probe_axis


def main() -> None:
    """Plot validation AP and rainy F1 for each architecture and seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/m6_4_1"))
    parser.add_argument("--output", type=Path, default=Path("docs/experiments/artifacts/M6.4.1/pooled-weather.png"))
    args = parser.parse_args()
    results = json.loads((args.results_dir / "results.json").read_text())
    arms = ("dense", *PLOT_ARMS)
    labels = ["Dense MAE", *(LABELS[arm] for arm in PLOT_ARMS)]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.1), sharey=True)
    fig.subplots_adjust(left=0.27, right=0.98, top=0.77, bottom=0.15, wspace=0.14)
    fig.suptitle("Pooled weather probe on BDD100K validation", x=0.27, ha="left", fontsize=18, weight="bold", color=INK)
    handles = [
        Line2D([], [], marker="o", ls="none", color="#8EA1B1", label="Model seed"),
        Line2D([], [], marker="o", color=INK, label="Mean and 95% interval"),
    ]
    fig.legend(handles=handles, ncols=2, loc="upper left", bbox_to_anchor=(0.265, 0.91), frameon=False)
    for ax, metric, title in zip(
        axes, ("average_precision", "rainy_f1"), ("Average precision", "Rainy-class F1"), strict=True
    ):
        style_probe_axis(ax, labels)
        ax.set_title(title, loc="left", weight="bold", color=INK, pad=12)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.axhline(0.5, color="#CBD8DC", lw=1)
        endpoints = []
        for y, arm in enumerate(arms):
            scores = [100 * row["scores"][metric] for row in results["runs"] if row["arm"] == arm]
            summary = results["summary"][arm][metric]
            lo, hi = (100 * p for p in summary["ci95"])
            color = WITHIN if arm == "dense" else INK
            ax.scatter(scores, y + np.linspace(-0.18, 0.18, len(scores)), color="#8EA1B1", s=20, zorder=3)
            ax.hlines(y, lo, hi, color=color, lw=2, zorder=4)
            ax.scatter(100 * summary["mean"], y, color=color, s=58, edgecolor="white", linewidth=0.7, zorder=5)
            endpoints.extend([*scores, lo, hi])
        ax.set_xlim(max(0, min(endpoints) - 2), min(100, max(endpoints) + 2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
