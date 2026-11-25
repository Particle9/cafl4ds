"""Additional uncapped GMM arm, compared with the frozen capped M6.2.0 arms."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from cafl4ds.cluster_routing_analysis import _interval, read_rows, run_metrics
from cafl4ds.expert_lr_analysis import (
    _time_mean,
    _verify_pair,
    audit_learning_rates,
    parameter_displacements,
    validate_baseline,
)
from cafl4ds.routing_experiment import digest_file, write_json

if TYPE_CHECKING:
    from matplotlib.figure import Figure

CONDITIONS = {
    "learned_capped": ("Learned · capped", "#2878b5"),
    "gmm_capped": ("GMM · capped", "#e07a28"),
    "gmm_uncapped": ("GMM · 4q", "#429b62"),
}


def condition_folders(output: Path, baseline: Path, seed: int) -> dict[str, Path]:
    """Resolve existing capped arms and the separately stored added arm."""
    return {
        "learned_capped": baseline / f"seed_{seed}/learned",
        "gmm_capped": baseline / f"seed_{seed}/gmm",
        "gmm_uncapped": output / f"seed_{seed}/gmm",
    }


def _audit_fixed_trace(output: Path, baseline: Path, seed: int) -> int:
    new_folder, old_folder = output / f"seed_{seed}/gmm", baseline / f"seed_{seed}/gmm"
    original, current = (json.loads((folder / "complete.json").read_text()) for folder in (old_folder, new_folder))
    _verify_pair(old_folder, new_folder, original, current)
    if current["trainable_router_parameters"] != 0:
        raise ValueError("The additional fixed GMM arm contains learned router parameters.")
    old, new = (read_rows(folder / "training.jsonl") for folder in (old_folder, new_folder))
    for left, right in zip(old, new, strict=True):
        if right["auxiliary"] != 0 or right["weighted_auxiliary"] != 0:
            raise ValueError("Unexpected GMM auxiliary loss.")
        for key, value in left.items():
            if key.endswith(("_mean_probabilities", "_mass_by_regime", "_batch_top1_expert", "_image_top1_counts")):
                if value != right[key]:
                    raise ValueError("Fixed routing differs from the original GMM trace.")
    for old_path in sorted(old_folder.glob("panel_*.npz")):
        old_arrays, new_arrays = np.load(old_path), np.load(new_folder / old_path.name)
        for key in old_arrays.files:
            if key.endswith("_probabilities"):
                np.testing.assert_array_equal(old_arrays[key], new_arrays[key])
    return len(new)


def analyze_uncapped(output: Path, baseline: Path) -> dict[str, Any]:
    """Audit the added arm and pair its metrics against each original M6.2.0 arm."""
    protocol = json.loads((output / "protocol.json").read_text())
    verification = validate_baseline(protocol, baseline)
    if protocol["config"].get("expert_lr_cap", 1) is not None or protocol["config"].get("arms") != ["gmm"]:
        raise ValueError("Expected the additional uncapped GMM arm only.")
    for name, fingerprint in protocol["source_hashes"].items():
        source = Path(__file__).parent / name
        if digest_file(source) != fingerprint or digest_file(output / "provenance" / name) != fingerprint:
            raise ValueError(f"Frozen source changed: {name}")
    steps = sum(_audit_fixed_trace(output, baseline, int(seed)) for seed in protocol["config"]["seeds"])
    lr_audit = audit_learning_rates(output)
    displacements = parameter_displacements(output, baseline)
    for row in displacements["records"]:
        row["capped_displacement"] = row.pop("constant_lr_displacement")
        row["uncapped_displacement"] = row.pop("expert_lr_displacement")
    write_json(output / "expert_displacement.json", displacements)
    summaries: dict[tuple[int, str], dict[str, float]] = {}
    for seed in protocol["config"]["seeds"]:
        for condition, folder in condition_folders(output, baseline, int(seed)).items():
            complete = json.loads((folder / "complete.json").read_text())
            metrics = run_metrics(complete)
            rows = read_rows(folder / "evaluation.jsonl")
            metrics["time_mean_mse_pass_1"] = _time_mean(rows, complete["steps_per_epoch"])
            metrics["time_mean_mse_full"] = _time_mean(rows, complete["steps"])
            summaries[(int(seed), condition)] = metrics
    keys = sorted(set.intersection(*(set(row) for row in summaries.values())))
    stats = {
        condition: {
            key: _interval([summaries[(int(seed), condition)][key] for seed in protocol["config"]["seeds"]])
            for key in keys
        }
        for condition in CONDITIONS
    }
    effects: dict[str, Any] = {}
    for control in ("gmm_capped", "learned_capped"):
        effects[control] = {}
        for key in keys:
            differences = [
                summaries[(int(seed), "gmm_uncapped")][key] - summaries[(int(seed), control)][key]
                for seed in protocol["config"]["seeds"]
            ]
            effects[control][key] = {**_interval(differences), "seed_differences": differences}
    result = {
        "experiment": "M6.2.0 additional GMM arm",
        "rule": "s_e = 4*q_e; lr_e = 0.001*s_e; no cap",
        "seeds": protocol["config"]["seeds"],
        "verification": {
            **verification,
            "matched_every_initial_tensor": displacements["all_saved_initial_tensors_match_baseline_exactly"],
            "matched_every_training_image_order_view_and_mask": True,
            "matched_every_fixed_routing_readout_and_evaluation_panel": True,
            "audited_added_gradient_steps": steps,
        },
        "learning_rate_audit": lr_audit,
        "conditions": stats,
        "uncapped_minus": effects,
        "replicate_unit": "paired training seed",
    }
    write_json(output / "comparison.json", result)
    (output / "analysis").mkdir(exist_ok=True)
    import matplotlib.pyplot as plt  # noqa: PLC0415

    figure = plot_conditions(output, baseline)
    figure.savefig(output / "analysis/three_arm_learning.png", dpi=150)
    plt.close(figure)
    return result


def plot_conditions(output: Path, baseline: Path, metric: str | None = None) -> Figure:
    """Compare all three arms on common evaluation steps with seed SEM bands."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    protocol = json.loads((output / "protocol.json").read_text())
    keys = (
        [metric] if metric else ["heldout_mse", "rankme", "balanced_linear", "entropy_mean", "alignment", "cka_drift"]
    )
    fig, axes = plt.subplots(
        1 if metric else 2,
        1 if metric else 3,
        figsize=(13, 4) if metric else (15, 8),
        constrained_layout=True,
        squeeze=False,
    )
    for condition, (label, color) in CONDITIONS.items():
        traces = [
            read_rows(condition_folders(output, baseline, int(seed))[condition] / "evaluation.jsonl")
            for seed in protocol["config"]["seeds"]
        ]
        for ax, key in zip(axes.flat, keys, strict=True):
            maps = [{row["step"]: row[key] for row in trace if key in row} for trace in traces]
            common = sorted(set.intersection(*(set(row) for row in maps)))
            matrix = np.array([[row[step] for step in common] for row in maps])
            mean = matrix.mean(0)
            sem = matrix.std(0, ddof=1) / np.sqrt(len(matrix)) if len(matrix) > 1 else np.zeros_like(mean)
            ax.plot(common, mean, label=label, color=color)
            ax.fill_between(common, mean - sem, mean + sem, color=color, alpha=0.12)
            ax.set(title=key, xlabel="Gradient update")
            ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("M6.2.0 · three arms · matched seeds · mean ±1 SEM")
    return fig


def plot_added_regimes(output: Path, baseline: Path, seed: int, change: bool = False) -> Figure:
    """Plot all three regime heatmaps with a common scale and the existing pre-exit masking."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    names = json.loads((output / "membership.json").read_text())["regime_names"]
    regimes = sorted(map(int, names))
    panels = []
    for condition, folder in condition_folders(output, baseline, seed).items():
        rows = read_rows(folder / "evaluation.jsonl")
        values = np.array([[row["regime_mse"][str(regime)] for row in rows] for regime in regimes])
        if change:
            values[:] = np.nan
            complete = json.loads((folder / "complete.json").read_text())
            for entry in complete["forgetting"]:
                for index, row in enumerate(rows):
                    if entry["exit_step"] <= row["step"] <= entry["pass_end_step"]:
                        values[regimes.index(entry["regime"]), index] = (
                            row["regime_mse"][str(entry["regime"])] - entry["mse_at_exit"]
                        )
        panels.append((condition, np.array([row["step"] for row in rows]), values))
    valid = np.concatenate([values[np.isfinite(values)] for _, _, values in panels])
    limit = float(abs(valid).max()) if len(valid) else 1
    bounds = (-limit, limit) if change else (float(valid.min()), float(valid.max()))
    fig, axes = plt.subplots(3, 1, figsize=(15, max(9, len(regimes) * 0.75)), constrained_layout=True, sharex=True)
    for ax, (condition, steps, values) in zip(axes, panels, strict=True):
        edges = np.r_[steps[0], (steps[:-1] + steps[1:]) / 2, steps[-1] + 0.5]
        mesh = ax.pcolormesh(
            edges,
            np.arange(len(regimes) + 1),
            values,
            cmap="coolwarm" if change else "viridis",
            vmin=bounds[0],
            vmax=bounds[1],
            shading="flat",
        )
        ax.set_yticks(np.arange(len(regimes)) + 0.5, [names[str(r)] for r in regimes], fontsize=8)
        ax.invert_yaxis()
        ax.set(title=CONDITIONS[condition][0], xlabel="Gradient update")
    fig.colorbar(mesh, ax=list(axes), label="MSE change since block exit" if change else "Normalized masked MSE")
    return fig


def plot_three_rates(output: Path, baseline: Path, seed: int, layer: int) -> Figure:
    """Show rates, gradients, actual updates and minibatch top-1 for all three arms."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.colors import ListedColormap  # noqa: PLC0415

    fig, axes = plt.subplots(5, 3, figsize=(17, 13), constrained_layout=True, sharex=True, sharey="row")
    colors = [plt.get_cmap("tab10")(expert) for expert in range(4)]
    for column, (condition, folder) in enumerate(condition_folders(output, baseline, seed).items()):
        rows = read_rows(folder / "training.jsonl")
        steps = np.array([row["step"] for row in rows])
        for index, (key, label) in enumerate(
            (
                ("mean_probabilities", "Mean probability"),
                ("expert_lr", "Applied LR"),
                ("expert_grad_norm", "Unclipped gradient L2"),
                ("expert_update_norm", "Actual parameter-step L2"),
            )
        ):
            values = np.array([row[f"l{layer}_{key}"] for row in rows])
            for expert, color in enumerate(colors):
                axes[index, column].plot(
                    steps,
                    np.maximum(values[:, expert], 1e-16) if index >= 2 else values[:, expert],
                    color=color,
                    lw=0.65,
                    label=f"Expert {expert}",
                )
            axes[index, column].set(ylabel=label)
            if index >= 2:
                axes[index, column].set_yscale("log")
        top1 = np.array([row[f"l{layer}_batch_top1_expert"] for row in rows])
        axes[4, column].imshow(
            top1[None, :],
            aspect="auto",
            extent=(steps[0] - 0.5, steps[-1] + 0.5, -0.5, 0.5),
            cmap=ListedColormap(colors),
            vmin=0,
            vmax=3,
            interpolation="nearest",
        )
        axes[4, column].set(yticks=[], xlabel="Gradient update", ylabel="Batch top-1")
        axes[0, column].set_title(CONDITIONS[condition][0])
    axes[0, 0].legend(fontsize=8, ncols=2)
    fig.suptitle(f"Seed {seed} · layer {layer} · raw per-step readouts · log-plot zeros shown at 1e-16")
    return fig
