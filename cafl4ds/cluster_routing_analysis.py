"""Saved-result analysis and notebook plots for M6.1.0; seeds are the replicate unit."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import t as student_t

from cafl4ds.routing_experiment import write_json

if TYPE_CHECKING:
    from matplotlib.figure import Figure

ARMS = ("learned", "gmm")
COLORS = {"learned": "#2878b5", "gmm": "#e07a28"}


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read a committed scientific JSONL trace."""
    return [json.loads(line) for line in path.read_text().splitlines()]


def _interval(values: list[float]) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if len(x) > 1 else None
    half = float(student_t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x))) if sd is not None else None
    return {"mean": mean, "sd": sd, "ci95": None if half is None else [mean - half, mean + half], "n_seeds": len(x)}


def run_metrics(result: dict[str, Any]) -> dict[str, float]:
    """Reduce within each seed before constructing any ensemble interval."""
    metrics = {
        key: float(result["final"][key])
        for key in (
            "heldout_mse",
            "balanced_mse",
            "balanced_knn",
            "balanced_linear",
            "natural_knn",
            "natural_linear",
            "rankme",
            "entropy_mean",
            "alignment",
            "uniformity",
            "cka_drift",
            "cosine_drift",
        )
    }
    for epoch in sorted({row["epoch"] for row in result["forgetting"]}):
        rows = [row for row in result["forgetting"] if row["epoch"] == epoch and row["post_exit_updates"] > 0]
        if rows:
            metrics[f"pass_{epoch + 1}_end_minus_exit"] = float(np.mean([row["end_minus_exit"] for row in rows]))
            metrics[f"pass_{epoch + 1}_end_minus_best"] = float(
                np.mean([row["end_minus_best_after_exit"] for row in rows])
            )
    metrics["update_seconds"] = float(result["elapsed_seconds"]["update"])
    return metrics


def analyze(  # noqa: C901 - audit every paired scientific trace before aggregating the ensemble
    output: Path, results: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Verify every paired step, summarize endpoints and render the completed ensemble."""
    protocol = json.loads((output / "protocol.json").read_text())
    if results is None:
        results = [
            json.loads((output / f"seed_{seed}" / arm / "complete.json").read_text())
            for seed in protocol["config"]["seeds"]
            for arm in ARMS
        ]
    by_pair = {(row["seed"], row["arm"]): row for row in results}
    for seed in protocol["config"]["seeds"]:
        learned, fixed = (by_pair[(seed, arm)] for arm in ARMS)
        if learned["shared_initial_hash"] != fixed["shared_initial_hash"] or learned["steps"] != fixed["steps"]:
            raise ValueError("Paired arms have mismatched initial weights or update budgets.")
        if fixed["trainable_router_parameters"] != 0:
            raise ValueError("The fixed GMM arm unexpectedly contains router parameters.")
        left, right = (read_rows(output / f"seed_{seed}" / arm / "training.jsonl") for arm in ARMS)
        if len(left) != len(right) or len(left) != learned["steps"]:
            raise ValueError("Incomplete step traces.")
        for a, b in zip(left, right, strict=True):
            if any(a[key] != b[key] for key in ("step", "order_hash", "views_hash", "mask_noise_hash", "sample_ids")):
                raise ValueError("Paired step order, augmentation or masking differs.")
            for row in (a, b):
                for key in row:
                    if key.endswith("_mean_probabilities"):
                        layer = key.removesuffix("_mean_probabilities")
                        if row[layer + "_batch_top1_expert"] != int(np.argmax(row[key])):
                            raise ValueError("Step top-1 expert is not argmax of minibatch mean probabilities.")
            if b["auxiliary"] != 0:
                raise ValueError("Fixed routing unexpectedly contributes an auxiliary training loss.")
    summaries = [{"seed": row["seed"], "arm": row["arm"], **run_metrics(row)} for row in results]
    metrics = set.intersection(*(set(row) - {"seed", "arm"} for row in summaries))
    arm_stats = {
        arm: {metric: _interval([row[metric] for row in summaries if row["arm"] == arm]) for metric in sorted(metrics)}
        for arm in ARMS
    }
    paired = {}
    for metric in sorted(metrics):
        values = []
        for seed in protocol["config"]["seeds"]:
            a = next(row for row in summaries if row["seed"] == seed and row["arm"] == "learned")
            b = next(row for row in summaries if row["seed"] == seed and row["arm"] == "gmm")
            values.append(b[metric] - a[metric])
        paired[metric] = {**_interval(values), "seed_differences": values}
    comparison = {
        "experiment": protocol["experiment"],
        "replicate_unit": "training seed",
        "runs": results,
        "arms": arm_stats,
        "gmm_minus_learned": paired,
        "verification": {
            "matched_shared_initialization": True,
            "matched_every_training_image_order": True,
            "matched_every_augmentation_and_mask": True,
            "gmm_has_no_router_parameters": True,
            "step_top1_is_argmax_of_minibatch_mean_probability": True,
        },
        "interpretation": (
            "Descriptive paired 95% t intervals over seeds; validation frames and checkpoints are not replicates."
        ),
    }
    write_json(output / "comparison.json", comparison)
    analysis = output / "analysis"
    analysis.mkdir(exist_ok=True)
    with (analysis / "seed_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seed", "arm", *sorted(metrics)])
        writer.writeheader()
        writer.writerows(summaries)
    for name, figure in (
        ("learning_curves", plot_learning(output)),
        ("regime_reconstruction", plot_regimes(output, int(protocol["config"]["seeds"][0]))),
        (
            "expert_timeline",
            plot_routing(
                output,
                int(protocol["config"]["seeds"][0]),
                min(
                    protocol["config"]["encoder"].get("moe_blocks")
                    or range(max(0, protocol["config"]["encoder"]["depth"] - 2), protocol["config"]["encoder"]["depth"])
                ),
            ),
        ),
        ("paired_endpoints", plot_paired(comparison)),
    ):
        figure.savefig(analysis / f"{name}.png", dpi=150)
        import matplotlib.pyplot as plt  # noqa: PLC0415 - optional analysis dependency

        plt.close(figure)
    return comparison


def plot_learning(output: Path) -> Figure:
    """Show seed trajectories, with ensemble means only at shared evaluation steps."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    protocol = json.loads((output / "protocol.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    keys = (
        ("heldout_mse", "Natural held-out reconstruction MSE"),
        ("balanced_linear", "Balanced scene linear probe"),
        ("rankme", "RankMe"),
        ("entropy_mean", "Clean image router entropy / log(4)"),
        ("alignment", "Fixed-view alignment"),
        ("cka_drift", "CKA drift from initialization"),
    )
    for arm in ARMS:
        traces = [read_rows(output / f"seed_{seed}" / arm / "evaluation.jsonl") for seed in protocol["config"]["seeds"]]
        for ax, (key, title) in zip(axes.flat, keys, strict=True):
            maps = [{row["step"]: row[key] for row in trace if key in row} for trace in traces]
            for values in maps:
                ax.plot(list(values), list(values.values()), color=COLORS[arm], alpha=0.2, lw=0.8)
            common = sorted(set.intersection(*(set(values) for values in maps)))
            matrix = np.array([[values[step] for step in common] for values in maps])
            mean = matrix.mean(0)
            sem = matrix.std(0, ddof=1) / np.sqrt(len(maps)) if len(maps) > 1 else np.zeros_like(mean)
            ax.plot(common, mean, color=COLORS[arm], label=arm)
            ax.fill_between(common, mean - sem, mean + sem, color=COLORS[arm], alpha=0.15)
            ax.set(title=title, xlabel="Gradient update")
            ax.grid(alpha=0.2)
    axes.flat[0].legend()
    fig.suptitle(f"{protocol['experiment']} · matched top-4 image routing · thin lines = seeds; shading = ±1 SEM")
    return fig


def plot_regimes(output: Path, seed: int, change_since_exit: bool = False) -> Figure:
    """Show true evaluation-step coordinates and a shared color scale across the two arms."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    membership = json.loads((output / "membership.json").read_text())
    names = membership["regime_names"]
    regimes = sorted(map(int, names))
    panels = []
    for arm in ARMS:
        rows = read_rows(output / f"seed_{seed}" / arm / "evaluation.jsonl")
        result = json.loads((output / f"seed_{seed}" / arm / "complete.json").read_text())
        values = np.array([[row["regime_mse"][str(regime)] for row in rows] for regime in regimes])
        if change_since_exit:
            values[:] = np.nan
            for entry in result["forgetting"]:
                for j, row in enumerate(rows):
                    if entry["exit_step"] <= row["step"] <= entry["pass_end_step"]:
                        values[regimes.index(entry["regime"]), j] = (
                            row["regime_mse"][str(entry["regime"])] - entry["mse_at_exit"]
                        )
        panels.append((np.array([row["step"] for row in rows]), values))
    all_values = np.concatenate([values[np.isfinite(values)] for _, values in panels])
    limit = float(np.abs(all_values).max()) if len(all_values) else 1.0
    vmin, vmax = (-limit, limit) if change_since_exit else (float(all_values.min()), float(all_values.max()))
    fig, axes = plt.subplots(2, 1, figsize=(15, max(7, len(regimes) * 0.55)), constrained_layout=True, sharex=True)
    for ax, arm, (steps, values) in zip(axes, ARMS, panels, strict=True):
        edges = np.r_[steps[0], (steps[:-1] + steps[1:]) / 2, steps[-1] + 0.5]
        mesh = ax.pcolormesh(
            edges,
            np.arange(len(regimes) + 1),
            values,
            cmap="coolwarm" if change_since_exit else "viridis",
            vmin=vmin,
            vmax=vmax,
            shading="flat",
        )
        ax.set_yticks(np.arange(len(regimes)) + 0.5, [names[str(regime)] for regime in regimes], fontsize=8)
        ax.invert_yaxis()
        ax.set(title=f"{arm} · seed {seed}", xlabel="Gradient update")
    fig.colorbar(
        mesh,
        ax=list(axes),
        label="MSE change since this pass's block exit" if change_since_exit else "Normalized masked MSE",
    )
    return fig


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    window = min(window, len(values))
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_routing(output: Path, seed: int, layer: int) -> Figure:
    """Align regime blocks, minibatch top-1, soft expert mass and actual expert gradients."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.colors import ListedColormap  # noqa: PLC0415

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(16, 9),
        sharex=True,
        sharey="row",
        constrained_layout=True,
        gridspec_kw={"height_ratios": [0.4, 0.4, 2, 2]},
    )
    expert_colors = [plt.get_cmap("tab10")(i) for i in range(4)]
    for column, arm in enumerate(ARMS):
        rows = read_rows(output / f"seed_{seed}" / arm / "training.jsonl")
        steps = np.array([row["step"] for row in rows])
        axes[0, column].imshow(
            [[row["dominant_regime"] for row in rows]],
            aspect="auto",
            extent=(0.5, len(rows) + 0.5, 0, 1),
            cmap="tab20",
            vmin=0,
            vmax=19,
            interpolation="nearest",
        )
        axes[1, column].imshow(
            [[row[f"l{layer}_batch_top1_expert"] for row in rows]],
            aspect="auto",
            extent=(0.5, len(rows) + 0.5, 0, 1),
            cmap=ListedColormap(expert_colors),
            vmin=-0.5,
            vmax=3.5,
            interpolation="nearest",
        )
        axes[0, column].set(title=f"{arm} · seed {seed} · layer {layer}", ylabel="Regime")
        axes[1, column].set(ylabel="Top-1")
        axes[0, column].set_yticks([])
        axes[1, column].set_yticks([])
        regime_ids = np.array([row["dominant_regime"] for row in rows])
        starts = np.r_[0, np.flatnonzero(np.diff(regime_ids)) + 1]
        stops = np.r_[starts[1:], len(rows)]
        for start, stop in zip(starts, stops, strict=True):
            if stop - start >= max(1, len(rows) // 100):
                axes[0, column].text(
                    (start + stop + 1) / 2,
                    0.5,
                    str(regime_ids[start]),
                    ha="center",
                    va="center",
                    fontsize=7,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 0.5},
                )
        mass = np.array([row[f"l{layer}_mean_probabilities"] for row in rows])
        gradients = np.array([row[f"l{layer}_expert_grad_norm"] for row in rows])
        for expert, color in enumerate(expert_colors):
            axes[2, column].plot(steps, mass[:, expert], color=color, lw=0.9, label=f"Expert {expert}")
            smoothed = _smooth(gradients[:, expert], 25)
            axes[3, column].plot(
                steps[-len(smoothed) :],
                np.where(smoothed > 0, smoothed, np.nan),
                color=color,
                lw=1,
                marker="." if len(smoothed) == 1 else None,
            )
        axes[2, column].set(ylabel="Mean image probability", ylim=(-0.02, 1.02))
        axes[3, column].set(
            ylabel="Unclipped gradient L2\n25-update moving mean", yscale="log", xlabel="Gradient update"
        )
        changes = np.flatnonzero(np.diff([row["dominant_regime"] for row in rows])) + 1.5
        for ax in axes[:, column]:
            for boundary in changes:
                ax.axvline(boundary, color="gray", alpha=0.18, lw=0.5)
    axes[2, 0].legend(ncols=4, fontsize=8, loc="upper center")
    fig.suptitle("Top-1 = argmax of minibatch mean probabilities; every update evaluates all four experts")
    return fig


def plot_paired(comparison: dict[str, Any]) -> Figure:
    """Show the actual seed pairs rather than treating checkpoint counts as replication."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    keys = [
        ("heldout_mse", "Final natural reconstruction MSE"),
        ("pass_1_end_minus_exit", "Pass 1 mean regime deterioration"),
        ("pass_2_end_minus_exit", "Pass 2 mean regime deterioration"),
        ("balanced_linear", "Final balanced linear probe"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    seeds = sorted({row["seed"] for row in comparison["runs"]})
    by_pair = {(row["seed"], row["arm"]): run_metrics(row) for row in comparison["runs"]}
    for ax, (key, title) in zip(axes, keys, strict=True):
        for seed in seeds:
            if key in by_pair[(seed, "learned")] and key in by_pair[(seed, "gmm")]:
                ax.plot([0, 1], [by_pair[(seed, arm)][key] for arm in ARMS], marker="o", lw=1, label=f"Seed {seed}")
        ax.set(xticks=[0, 1], xticklabels=list(ARMS), title=title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    return fig
