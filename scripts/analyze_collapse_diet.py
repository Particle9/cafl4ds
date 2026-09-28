"""Offline P1.4.0 manifest, paired summaries, and publication-ready diagnostic figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cafl4ds_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "cafl4ds_cache"))

import matplotlib  # noqa: E402 - cache paths must be selected before importing Matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - backend must be set first
import numpy as np  # noqa: E402 - keep plotting imports together after backend setup

from cafl4ds.jsonio import dumps_valid
from cafl4ds.run_log import read_run


def _arms(root: Path) -> list[dict[str, Any]]:
    """Read complete canonical arm artifacts without loading weights or images."""
    arms = []
    for path in sorted(root.glob("*/comparison.json")):
        comparison = json.loads(path.read_text())
        if not comparison["gate"]["passed"]:
            continue
        extension = comparison["collapse_diet"]
        arms.append(
            {
                "name": path.parent.name,
                "path": str(path),
                "profile": extension["profile"],
                "seed": extension["provenance"]["seed"],
                "ordering": extension["provenance"]["ordering"],
                "policy": extension["provenance"]["policy"],
                "pc": extension["pc"],
                "comparison": comparison,
            }
        )
    return arms


def _trajectories(arms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain seed-level checkpoint values; query images are never treated as replicates."""
    rows = []
    for arm in arms:
        role = "pc" if arm["pc"] else "live"
        for point in arm["comparison"]["arms"][role]["health"]:
            rows.append(
                {
                    "name": arm["name"],
                    "profile": arm["profile"],
                    "seed": arm["seed"],
                    "ordering": arm["ordering"],
                    "policy": arm["policy"],
                    "pc": arm["pc"],
                    "optimizer_step": point.get("optimizer_step"),
                    "epoch": point.get("epoch"),
                    "rankme_proj": point.get("rankme_proj"),
                    "linear_acc": point.get("linear_acc"),
                    "knn_acc": point.get("knn_acc"),
                }
            )
    return rows


def _budget_traces(arms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Read realized batch, replay, and LR clocks for a representative seed per policy."""
    rows = []
    representatives: dict[str, dict[str, Any]] = {}
    for arm in arms:
        if not arm["pc"] and arm["profile"] == "matched_384" and arm["ordering"] == "iid":
            current = representatives.get(arm["policy"])
            if current is None or arm["seed"] < current["seed"]:
                representatives[arm["policy"]] = arm
    if not representatives:
        for arm in arms:
            if not arm["pc"] and arm["ordering"] == "iid":
                representatives.setdefault(arm["policy"], arm)
    for policy, arm in representatives.items():
        records = read_run(Path(arm["path"]).parent / f"{arm['name']}.jsonl")
        selection = {row["step"]: row for row in records if row["series"] == "selection"}
        for record in records:
            if record["series"] != "loss":
                continue
            decision = selection[record["step"]]
            rows.append(
                {
                    "policy": policy,
                    "seed": arm["seed"],
                    "step": record["step"],
                    "raw_count": decision["raw_count"],
                    "trained": record.get("trained"),
                    "current_count": len(decision["current_rows"]),
                    "replay_count": len(decision["replay_events"]),
                    "lr": record.get("lr"),
                }
            )
    return rows


def _save_trajectories(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """Plot each seed's IID/stress trajectory with a distinct ordering color."""
    if not rows:
        return
    profiles = ["matched_384", "legacy_400", "smoke"]
    profile = next((name for name in profiles if any(row["profile"] == name for row in rows)), profiles[-1])
    selected = [row for row in rows if row["profile"] == profile and row["policy"] == "accept_all" and not row["pc"]]
    colors = {"iid": "#2364aa", "b128": "#3da35d", "b256": "#e89b2c", "full": "#a23b72"}
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    for ordering, seed in sorted({(row["ordering"], row["seed"]) for row in selected}):
        points = sorted(
            (row for row in selected if row["ordering"] == ordering and row["seed"] == seed),
            key=lambda row: row["optimizer_step"],
        )
        for ax, key in zip(axes, ("rankme_proj", "linear_acc"), strict=True):
            available = [row for row in points if row[key] is not None]
            ax.plot(
                [row["optimizer_step"] for row in available],
                [row[key] for row in available],
                color=colors.get(ordering, "gray"),
                alpha=0.25,
                marker="." if key == "linear_acc" else None,
            )
    for ordering in ("iid", "b128", "b256", "full"):
        series = [row for row in selected if row["ordering"] == ordering]
        if not series:
            continue
        for ax, key in zip(axes, ("rankme_proj", "linear_acc"), strict=True):
            steps = sorted({row["optimizer_step"] for row in series if row[key] is not None})
            means = [
                np.mean([row[key] for row in series if row["optimizer_step"] == step and row[key] is not None])
                for step in steps
            ]
            ax.plot(steps, means, color=colors[ordering], linewidth=2.5, label=ordering)
    axes[0].set_ylabel("Projector RankMe")
    axes[1].set_ylabel("Ten-way linear accuracy")
    axes[1].set_xlabel("Optimizer updates")
    axes[0].legend(ncol=4, loc="best")
    axes[0].set_title(f"P1.4.0: intact SimSiam on {profile} (thin lines: seeds)")
    fig.savefig(out_dir / "trajectories.png", dpi=180)
    plt.close(fig)


def _save_effects(summary: dict[str, Any], out_dir: Path, *, filename: str) -> None:
    """Plot paired seed effects and bootstrap intervals without pooling checkpoints."""
    if "cells" in summary:
        entries = [(name, item["quality"], item["geometry"]) for name, item in summary["cells"].items()]
        title = "Stress dose vs matched IID"
    elif "effects" in summary:
        entries = [(name, item["quality"], item["geometry"]) for name, item in summary["effects"].items()]
        title = "Policy benefit vs matched budget control"
    else:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for ax, offset, label in zip(
        axes,
        (100.0, 100.0),
        ("Linear accuracy difference (percentage points)", "Projector RankMe log ratio ×100"),
        strict=True,
    ):
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(range(len(entries)), [name for name, _, _ in entries])
        ax.set_xlabel(label)
        for idx, (_, quality, geometry) in enumerate(entries):
            effect = quality if ax is axes[0] else geometry
            data = np.array(effect["by_seed"], dtype=float) * offset
            ax.scatter(data, np.full(len(data), idx), color="#718096", alpha=0.7, s=23)
            mean = float(effect["mean"]) * offset
            lo, hi = (float(v) * offset for v in effect["ci95"])
            ax.errorbar(mean, idx, xerr=[[mean - lo], [hi - mean]], color="#174a8b", fmt="D", capsize=4)
    fig.suptitle(title)
    fig.savefig(out_dir / filename, dpi=180)
    plt.close(fig)


def _save_budgets(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """Show realized admission/replay quotas and the update-level LR sequence."""
    if not rows:
        return
    policies = sorted({row["policy"] for row in rows})
    colors = {name: plt.get_cmap("tab10")(index) for index, name in enumerate(policies)}
    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
    for policy in policies:
        points = sorted((row for row in rows if row["policy"] == policy), key=lambda row: row["step"])
        steps = [row["step"] for row in points]
        axes[0].plot(steps, [row["trained"] for row in points], label=policy, color=colors[policy])
        axes[1].plot(
            steps,
            [row["current_count"] / row["raw_count"] for row in points],
            label=policy,
            color=colors[policy],
        )
        axes[2].plot(steps, [row["replay_count"] for row in points], label=policy, color=colors[policy])
        if all(row["lr"] is not None for row in points):
            axes[3].plot(steps, [row["lr"] for row in points], label=policy, color=colors[policy])
    for ax, label in zip(
        axes,
        ("Trained images/update", "Current admitted / raw", "Replay images/update", "Learning rate"),
        strict=True,
    ):
        ax.set_ylabel(label)
    axes[0].legend(ncol=4)
    axes[-1].set_xlabel("Arrival / optimizer step")
    fig.suptitle("Realized budgets and schedule (one seed per policy)")
    fig.savefig(out_dir / "budget_traces.png", dpi=180)
    plt.close(fig)


def analyze(root: Path, out_dir: Path) -> dict[str, Any]:
    """Build a compact manifest, outcome record, figure data, and four report plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = _arms(root)
    trajectories = _trajectories(arms)
    budgets = _budget_traces(arms)
    stages = {
        name: json.loads(path.read_text())
        for name in ("smoke", "bridge", "susceptibility", "selection", "confirmation")
        if (path := root / f"{name}_summary.json").exists()
    }
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else None
    confirmed = stages.get("confirmation")
    development = stages.get("susceptibility")
    summary = {
        "analysis_provenance": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "matplotlib": matplotlib.__version__,
        },
        "apparatus_valid": confirmed.get("apparatus_valid") if confirmed else None,
        "healthy_learning_established": (
            confirmed["references"]["accept_all"]["confirmation_learning_pass"]
            if confirmed
            else development["reference"]["development_learning_pass"]
            if development
            else None
        ),
        "diet_quality_effect": confirmed["diet"]["quality"] if confirmed else None,
        "diet_geometry_effect": confirmed["diet"]["geometry"] if confirmed else None,
        "collapse_interpretation": confirmed["diet"]["classification"] if confirmed else "not_evaluated",
        "admission_effect": confirmed["effects"]["admission"] if confirmed else None,
        "replay_effect": confirmed["effects"]["replay"] if confirmed else None,
        "protection_verdict": confirmed["protection_verdict"] if confirmed else None,
        "single_pass_transfer_status": "not_run",
        "scope_limitations": ["STL-10 labelled train split", "from-scratch SimSiam", "CPU", "fixed split by seed"],
        "stage_summaries": list(stages),
        "protocol_hash": protocol["hash"] if protocol else None,
        "completed_arms": len(arms),
    }
    manifest = [
        {key: arm[key] for key in ("name", "path", "profile", "seed", "ordering", "policy", "pc")}
        | {
            "provenance_hash": arm["comparison"]["collapse_diet"]["provenance"]["hash"],
            "split_hash": arm["comparison"]["collapse_diet"]["provenance"]["split_hash"],
            "initial_state_hash": arm["comparison"]["collapse_diet"]["initial_state_hash"],
        }
        for arm in arms
    ]
    (out_dir / "study_manifest.json").write_text(dumps_valid(manifest), encoding="utf-8")
    (out_dir / "study_summary.json").write_text(dumps_valid(summary), encoding="utf-8")
    if protocol is not None:
        (out_dir / "protocol.json").write_text(dumps_valid(protocol), encoding="utf-8")
    (out_dir / "figure_data.json").write_text(
        dumps_valid({"trajectories": trajectories, "budgets": budgets, "stage_summaries": stages}), encoding="utf-8"
    )
    _save_trajectories(trajectories, out_dir)
    _save_budgets(budgets, out_dir)
    if development:
        _save_effects(development, out_dir, filename="dose_effects.png")
    if confirmed or "selection" in stages:
        _save_effects(confirmed or stages["selection"], out_dir, filename="policy_effects.png")
    return summary


def main() -> None:
    """Rebuild evidence from complete arm artifacts; never retrain for analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/p140"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out.resolve() if args.out else root / "report"
    result = analyze(root, out_dir)
    print(
        dumps_valid(
            {
                "report": str(out_dir),
                "completed_arms": result["completed_arms"],
                "collapse_interpretation": result["collapse_interpretation"],
            }
        )
    )


if __name__ == "__main__":
    main()
