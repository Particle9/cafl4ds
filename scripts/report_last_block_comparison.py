"""Audit trained parameter groups and plot completed last-block comparison artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def audit(out_dir: Path) -> dict[str, Any]:
    """Verify unchanged frozen tensors and actual changes in each intended trainable group."""
    protocol = json.loads((out_dir / "protocol.json").read_text(encoding="utf-8"))["payload"]
    well = Path(protocol["config"]["well"])
    if hashlib.sha256(well.read_bytes()).hexdigest() != protocol["checkpoint_sha256"]:
        raise ValueError("Warm checkpoint has changed since the study started.")
    initial = torch.load(well, map_location="cpu", weights_only=True)
    result = {}
    for seed in protocol["config"]["seeds"]:
        final = torch.load(
            out_dir / f"seed_{seed}/checkpoints/adapted_s{seed}.pt", map_location="cpu", weights_only=True
        )
        if final.keys() != initial.keys() or any(not torch.isfinite(v).all() for v in final.values()):
            raise ValueError(f"Invalid checkpoint for seed {seed}.")
        changed = [key for key in initial if not torch.equal(initial[key], final[key])]
        groups = ("encoder.blocks.3.", "encoder.norm.", "decoder.")
        if any(not key.startswith(groups) for key in changed) or not all(
            any(key.startswith(g) for key in changed) for g in groups
        ):
            raise ValueError(f"Unexpected frozen/trained parameter changes in seed {seed}.")
        result[str(seed)] = {
            "early_encoder_bitwise_unchanged": True,
            "changed_tensors": changed,
            "all_trainable_groups_changed": True,
        }
    (out_dir / "weight_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    """Audit final checkpoints and render paired means/intervals from the completed summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    report = json.loads(args.summary.read_text(encoding="utf-8"))
    audit(args.summary.parent)
    contrasts = ("last_minus_full", "last_minus_frozen", "full_minus_frozen")
    labels = ("Last block − Full", "Last block − Frozen", "Full − Frozen")
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    for ax, key, title in zip(
        axes,
        ("knn", "linear", "final_conditions"),
        ("Global kNN", "Global linear probe", "Mean final condition"),
        strict=True,
    ):
        values = [report["metrics"][contrast][key] for contrast in contrasts]
        means = np.array([v["mean_pp"] for v in values])
        low, high = np.array([v["ci95_pp"] for v in values]).T
        ax.axhline(0, color="0.4", linewidth=1)
        ax.errorbar(range(3), means, yerr=[means - low, high - means], fmt="D", capsize=5)
        ax.set_xticks(range(3), labels, rotation=30, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.set_xlim(-0.3, 2.3)
    axes[0].set_ylabel("Accuracy difference (pp), mean and 95% t interval")
    fig.suptitle("BDD100K last-block adaptation: five-seed development comparison")
    fig.text(
        0.5,
        0.01,
        "Three panels averaged per seed; intervals conditional on one checkpoint and validation panel set.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    destination = args.summary.with_name("comparison.png")
    fig.savefig(destination, dpi=160)
    print(f"Checkpoint audit passed; plot saved to {destination}")


if __name__ == "__main__":
    main()
