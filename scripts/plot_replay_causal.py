"""Plot the paired replay-versus-no-replay causal ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Render causal replay effects with paired five-seed intervals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    report = json.loads(args.summary.read_text(encoding="utf-8"))
    arms = ["replay_minus_no_replay", "replay_minus_frozen", "no_replay_minus_frozen"]
    labels = {"knn": "Global kNN", "linear": "Global linear", "final_conditions": "Final condition"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharey=True)
    for ax, metric in zip(axes, labels, strict=True):
        positions = np.arange(len(arms))
        means = [report["metrics"][arm][metric]["mean_pp"] for arm in arms]
        lows = [report["metrics"][arm][metric]["ci95_pp"][0] for arm in arms]
        highs = [report["metrics"][arm][metric]["ci95_pp"][1] for arm in arms]
        ax.axhline(0, color="0.4", linewidth=1)
        ax.errorbar(positions, means, yerr=[np.array(means) - lows, np.array(highs) - means], fmt="D", capsize=5)
        ax.set_xticks(positions, ["Replay−No replay", "Replay−Frozen", "No replay−Frozen"], rotation=35, ha="right")
        ax.set_title(labels[metric])
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Accuracy gain (percentage points)")
    fig.suptitle("BDD100K causal replay ablation", fontsize=13)
    fig.tight_layout()
    destination = args.summary.with_name("causal.png")
    fig.savefig(destination, dpi=160)
    print(destination)


if __name__ == "__main__":
    main()
