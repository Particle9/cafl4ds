"""Plot the fixed confirmation's five seed effects and conditional confidence intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Render an auditable comparison from the completed summary without rerunning training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    report = json.loads(args.summary.read_text(encoding="utf-8"))
    names = {"knn": "Global kNN", "linear": "Global linear probe", "final_conditions": "Mean final condition"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharey=True)
    for ax, (key, label) in zip(axes, names.items(), strict=True):
        metric = report["metrics"][key]
        values = metric["seed_gains_pp"]
        ax.axhline(0, color="0.4", linewidth=1)
        ax.scatter(np.arange(1, 6), values, label="Training seed", s=45, color="tab:blue")
        mean = metric["mean_pp"]
        low, high = metric["ci95_pp"]
        ax.errorbar(7, mean, yerr=[[mean - low], [high - mean]], fmt="D", color="tab:orange", capsize=5)
        ax.set_xticks([1, 2, 3, 4, 5, 7], [str(row["seed"]) for row in report["seeds"]] + ["Mean"], rotation=45)
        ax.set_title(label)
        ax.set_xlim(0, 8)
        ax.set_xlabel("Training seed / mean with 95% t interval")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Online replay minus frozen accuracy (pp)")
    fig.suptitle("BDD100K replay confirmation: unchanged training recipe, larger evaluation", fontsize=13)
    fig.text(
        0.5,
        0.015,
        "Three probe panels averaged per seed; intervals conditional on one warm checkpoint and dataset.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    destination = args.summary.with_name("confirmation.png")
    fig.savefig(destination, dpi=160)
    print(destination)


if __name__ == "__main__":
    main()
