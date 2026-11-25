"""Plot the saved M6.4.0 weather-probe results."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from cafl4ds.cluster_routing_analysis import _interval
from cafl4ds.day_night_analysis import LABELS
from cafl4ds.weather_probe import DOMAINS
from cafl4ds.weather_probe_plots import BASELINE, CROSS, GRID, INK, PLOT_ARMS, WITHIN, style_probe_axis


def _plot_f1(results: dict[str, Any], protocol: dict[str, Any], path: Path) -> None:
    summary = results["summary"]
    counts = Counter((row["timeofday"], row["weather"]) for row in protocol["panel"] if row["partition"] == "test")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.1), sharey=True)
    fig.subplots_adjust(left=0.27, right=0.98, top=0.78, bottom=0.15, wspace=0.13)
    fig.suptitle("Rainy F1 drops when the probe crosses domains", x=0.27, ha="left", fontsize=18, weight="bold")
    for ax, test, other in zip(axes, DOMAINS, reversed(DOMAINS), strict=True):
        style_probe_axis(ax, [LABELS[arm] for arm in PLOT_ARMS])
        ax.set_title(f"{test.capitalize()} test images", loc="left", weight="bold", pad=14)
        ax.set_xlim(0, 80)
        ax.set_xticks(range(0, 81, 20))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.set_xlabel("Rainy-class F1", labelpad=9)
        clear, rainy = counts[test, "clear"], counts[test, "rainy"]
        ax.axvline(200 * rainy / (clear + 2 * rainy), color=BASELINE, ls="--", lw=1.5)
        ax.axhline(1.5, color=GRID, lw=1)
        for y, arm in enumerate(PLOT_ARMS):
            scores = summary[arm]["scores"]
            within = 100 * scores[f"{test}_to_{test}"]["rainy_f1"]["mean"]
            cross = 100 * scores[f"{other}_to_{test}"]["rainy_f1"]["mean"]
            ax.hlines(y, cross, within, color="#9AA8B2", lw=1.7, zorder=2)
            ax.scatter(within, y, color=WITHIN, s=57, zorder=3)
            ax.scatter(cross, y, color=CROSS, marker="D", s=44, zorder=3)
    handles = [
        Line2D([], [], marker="o", ls="none", color=WITHIN, label="Trained in test domain"),
        Line2D([], [], marker="D", ls="none", color=CROSS, label="Trained in other domain"),
        Line2D([], [], ls="--", color=BASELINE, label="Always predicts rain"),
    ]
    fig.legend(handles=handles, ncols=3, loc="upper left", bbox_to_anchor=(0.265, 0.92), frameon=False)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_gaps(results: dict[str, Any], path: Path) -> None:
    gaps: dict[str, dict[int, float]] = {arm: {} for arm in PLOT_ARMS}
    for run in results["runs"]:
        gaps[run["arm"]][run["seed"]] = 100 * run["transfer_gap"]["rainy_f1"]
    contrasts = (
        ("Full trunks vs. split after two", "gmm_full", "gmm_split"),
        ("Private vs. shared decoder", "gmm_full_private_decoder", "gmm_full"),
    )
    fig, (ax, paired) = plt.subplots(2, 1, figsize=(11.8, 7.3), gridspec_kw={"height_ratios": [7, 2]})
    fig.subplots_adjust(left=0.27, right=0.98, top=0.83, bottom=0.11, hspace=0.76)
    fig.suptitle("Full trunks have the largest mean transfer gaps", x=0.27, ha="left", fontsize=18, weight="bold")
    style_probe_axis(ax, [LABELS[arm] for arm in PLOT_ARMS])
    ax.set_xlim(0, 45)
    ax.set_xticks(range(0, 46, 5))
    ax.set_xlabel("Within-domain minus cross-domain rainy F1 (points)", labelpad=9)
    ax.axhline(1.5, color=GRID, lw=1)
    style_probe_axis(paired, [label for label, _, _ in contrasts])
    paired.set_xlim(-12, 21)
    paired.set_xticks(range(-10, 21, 5))
    paired.axvline(0, color=BASELINE, lw=1.3, ls="--")
    paired.set_title("Paired architecture comparisons", loc="left", weight="bold", pad=10)
    paired.set_xlabel("Change in gap (points)", labelpad=9)
    values = [[gaps[arm][seed] - gaps[base][seed] for seed in sorted(gaps[base])] for _, arm, base in contrasts]
    for axis, rows in ((ax, [list(gaps[arm].values()) for arm in PLOT_ARMS]), (paired, values)):
        for y, seeds in enumerate(rows):
            stats = _interval(seeds)
            lo, hi = stats["ci95"]
            axis.scatter(seeds, y + np.linspace(-0.20, 0.20, len(seeds)), color="#8EA1B1", s=25, zorder=3)
            axis.hlines(y, lo, hi, color=INK, lw=2.1, zorder=4)
            axis.scatter(stats["mean"], y, color=INK, edgecolor="white", linewidth=0.7, s=70, zorder=5)
    handles = [
        Line2D([], [], marker="o", ls="none", color="#8EA1B1", label="Model seed"),
        Line2D([], [], marker="o", color=INK, label="Mean and 95% interval"),
    ]
    fig.legend(handles=handles, ncols=2, loc="upper left", bbox_to_anchor=(0.265, 0.92), frameon=False)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_dense_contrasts(small: dict[str, Any], large: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.1), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.27, right=0.98, top=0.75, bottom=0.16, wspace=0.14)
    fig.suptitle("Weather transfer gaps relative to the dense MAE", x=0.27, ha="left", fontsize=18, weight="bold")
    limit = max(
        abs(endpoint)
        for result in (small, large)
        for arm in PLOT_ARMS
        for endpoint in result["paired_f1_gap_vs_dense"][arm]["ci95"]
    )
    limit = max(5, int(np.ceil(100 * limit / 5)) * 5)
    for ax, result, label in zip(axes, (small, large), ("Pilot probes", "Larger probes"), strict=True):
        style_probe_axis(ax, [LABELS[arm] for arm in PLOT_ARMS])
        dense_gap = 100 * result["summary"]["dense"]["transfer_gap"]["rainy_f1"]["mean"]
        ax.set_title(f"{label}\nDense gap {dense_gap:.1f} points", loc="left", weight="bold", pad=12)
        ax.set_xlim(-limit, limit)
        ax.set_xticks(np.linspace(-limit, limit, 5))
        ax.set_xlabel("MoE gap minus dense gap (points)", labelpad=9)
        ax.axvline(0, color=BASELINE, ls="--", lw=1.3)
        ax.axhline(1.5, color=GRID, lw=1)
        for y, arm in enumerate(PLOT_ARMS):
            stats = result["paired_f1_gap_vs_dense"][arm]
            lo, hi = (100 * value for value in stats["ci95"])
            ax.hlines(y, lo, hi, color=INK, lw=2, zorder=2)
            ax.scatter(100 * stats["mean"], y, color=WITHIN, s=55, zorder=3)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    """Render report figures from protocol and results JSON files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/m6_4_0"))
    parser.add_argument("--output", type=Path, default=Path("docs/experiments/artifacts/M6.4.0"))
    args = parser.parse_args()
    results = json.loads((args.results_dir / "results.json").read_text())
    protocol = json.loads((args.results_dir / "protocol.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(
        {"font.size": 11, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK}
    ):
        _plot_f1(results, protocol, args.output / "rainy-f1.png")
        _plot_gaps(results, args.output / "transfer-gaps.png")
        dense = args.results_dir / "dense_control"
        if (dense / "results.json").is_file() and (dense / "train_2000/results.json").is_file():
            _plot_dense_contrasts(
                json.loads((dense / "results.json").read_text()),
                json.loads((dense / "train_2000/results.json").read_text()),
                args.output / "dense-control.png",
            )


if __name__ == "__main__":
    main()
