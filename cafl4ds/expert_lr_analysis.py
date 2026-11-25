"""M6.2.0 learning-rate audits and paired comparisons against the saved M6.1.0 baseline."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from cafl4ds.cluster_routing_analysis import ARMS, COLORS, _interval, read_rows, run_metrics
from cafl4ds.routing_experiment import digest_file, write_json

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def validate_baseline(protocol: dict[str, Any], baseline: Path) -> dict[str, Any]:
    """Refuse a comparison that changes anything besides the declared expert-LR intervention."""
    original = json.loads((baseline / "protocol.json").read_text())
    ignored = {"experiment_id", "expert_lr_scaling", "expert_lr_cap", "arms", "baseline_directory", "output"}
    old = {key: value for key, value in original["config"].items() if key not in ignored}
    new = {key: value for key, value in protocol["config"].items() if key not in ignored}
    if old != new:
        raise ValueError("Baseline scientific configuration differs beyond the expert-LR intervention.")
    for key in ("source", "conditioning", "panels", "runtime"):
        if json.loads(json.dumps(protocol[key])) != original[key]:
            raise ValueError(f"Baseline {key} changed.")
    for name, fingerprint in original["source_hashes"].items():
        if name.startswith(("models/", "ssl/")) or name in {
            "measurements.py",
            "routing_streams.py",
            "routing_experiment.py",
        }:
            if protocol["source_hashes"].get(name) != fingerprint:
                raise ValueError(f"Forward model, evaluation or data-order code changed: {name}")
    reference = {
        "directory": str(baseline),
        "protocol_sha256": digest_file(baseline / "protocol.json"),
        "comparison_sha256": digest_file(baseline / "comparison.json"),
        "matched_configuration_data_panels_runtime_and_forward_code": True,
    }
    if "baseline_reference" in protocol and protocol["baseline_reference"] != reference:
        raise ValueError("The frozen baseline protocol or comparison artifact changed.")
    return reference


def _time_mean(rows: list[dict[str, Any]], stop: int) -> float:
    selected = [row for row in rows if row["step"] <= stop]
    steps = np.array([row["step"] for row in selected])
    values = np.array([row["heldout_mse"] for row in selected])
    # Explicit trapezoids also work with the repository's minimum NumPy version.
    return float(((values[:-1] + values[1:]) * np.diff(steps) / 2).sum() / stop)


def _audit_row(row: dict[str, Any], base_lr: float, cap: float | None = 1.0) -> int:
    if row["shared_and_router_lr"] != base_lr:
        raise ValueError("A shared/router learning rate changed.")
    vectors = 0
    for key, scales in row.items():
        if not key.endswith("_expert_lr_scale"):
            continue
        prefix = key.removesuffix("_expert_lr_scale")
        mass = np.array(row[prefix + "_mean_probabilities"])
        expected = len(mass) * mass if cap is None else np.minimum(cap, len(mass) * mass)
        np.testing.assert_allclose(scales, expected, atol=1e-12, rtol=0)
        np.testing.assert_allclose(row[prefix + "_expert_lr"], base_lr * expected, atol=1e-12, rtol=0)
        changes = np.array(row[prefix + "_expert_update_norm"])
        if not np.isfinite(changes).all() or (changes < 0).any():
            raise ValueError("Invalid actual expert parameter-update norms.")
        vectors += 1
    return vectors


def audit_learning_rates(output: Path) -> dict[str, Any]:
    """Audit every applied LR and confirm Adam counters advance without inactive-step skipping."""
    protocol = json.loads((output / "protocol.json").read_text())
    base_lr = float(protocol["config"]["optim"]["lr"])
    cap = protocol["config"].get("expert_lr_cap", 1.0)
    updates, layer_records = 0, 0
    for seed in protocol["config"]["seeds"]:
        for arm in protocol["config"].get("arms", ARMS):
            folder = output / f"seed_{seed}" / arm
            rows = read_rows(folder / "training.jsonl")
            layer_records += sum(_audit_row(row, base_lr, cap=cap) for row in rows)
            checkpoint = torch.load(folder / "latest.pt", map_location="cpu", weights_only=False)
            optimizer = checkpoint["optimizer"]
            for group in optimizer["param_groups"]:
                if "expert_layer" in group:
                    for parameter in group["params"]:
                        if int(optimizer["state"][parameter]["step"]) != len(rows):
                            raise ValueError("An expert's AdamW step counter did not advance normally.")
            updates += len(rows)
    result = {
        "audited_gradient_steps": updates,
        "audited_layer_lr_vectors": layer_records,
        ("all_expert_rates_match_4q" if cap is None else "all_expert_rates_match_min_1_4q"): True,
        "shared_and_router_rates_unchanged": True,
        "all_expert_adam_counters_advanced_every_step": True,
        "all_actual_update_norms_finite": True,
    }
    write_json(output / "lr_verification.json", result)
    return result


def _verify_pair(old_folder: Path, new_folder: Path, original: dict[str, Any], current: dict[str, Any]) -> None:
    if original["shared_initial_hash"] != current["shared_initial_hash"] or original["steps"] != current["steps"]:
        raise ValueError("Cross-experiment initialization or budget mismatch.")
    old_rows, new_rows = (read_rows(folder / "training.jsonl") for folder in (old_folder, new_folder))
    for old, new in zip(old_rows, new_rows, strict=True):
        if any(old[key] != new[key] for key in ("sample_ids", "order_hash", "views_hash", "mask_noise_hash")):
            raise ValueError("Cross-experiment data, views or masks differ.")


def parameter_displacements(output: Path, baseline: Path) -> dict[str, Any]:
    """Verify all saved initial tensors and compare cumulative expert displacement at milestones."""
    protocol = json.loads((output / "protocol.json").read_text())
    records = []
    for seed in protocol["config"]["seeds"]:
        for arm in protocol["config"].get("arms", ARMS):
            folders = [root / f"seed_{seed}" / arm for root in (baseline, output)]
            initial = [
                torch.load(folder / "initial.pt", map_location="cpu", weights_only=False)["method"]
                for folder in folders
            ]
            if initial[0].keys() != initial[1].keys():
                raise ValueError("Cross-experiment initial parameter names differ.")
            for name in initial[0]:
                torch.testing.assert_close(initial[0][name], initial[1][name], rtol=0, atol=0)
            for path in sorted(folders[1].glob("checkpoint_*.pt")):
                states = [torch.load(folder / path.name, map_location="cpu", weights_only=False) for folder in folders]
                records.extend(_displacement_records(initial[0], states, int(seed), arm))
    result = {
        "all_saved_initial_tensors_match_baseline_exactly": True,
        "measure": "L2 distance from initialization, not accumulated path length",
        "records": records,
    }
    write_json(output / "expert_displacement.json", result)
    return result


def _displacement_records(
    initial: dict[str, torch.Tensor], states: list[dict[str, Any]], seed: int, arm: str
) -> list[dict[str, Any]]:
    squares: dict[tuple[int, int], list[float]] = {}
    for name, value in initial.items():
        match = re.search(r"blocks\.(\d+)\.mlp\.experts\.(\d+)\.", name)
        if match is None:
            continue
        key = (int(match[1]), int(match[2]))
        sums = squares.setdefault(key, [0.0, 0.0])
        for index, state in enumerate(states):
            sums[index] += float((state["method"][name] - value).double().square().sum())
    return [
        {
            "seed": seed,
            "arm": arm,
            "step": int(states[1]["step"]),
            "layer": layer,
            "expert": expert,
            "constant_lr_displacement": values[0] ** 0.5,
            "expert_lr_displacement": values[1] ** 0.5,
        }
        for (layer, expert), values in sorted(squares.items())
    ]


def compare_to_baseline(output: Path, baseline: Path) -> dict[str, Any]:
    """Check cross-experiment pairing and compute new-minus-original effects within each router."""
    protocol = json.loads((output / "protocol.json").read_text())
    verification = validate_baseline(protocol, baseline)
    learning_rate_audit = audit_learning_rates(output)
    displacements = parameter_displacements(output, baseline)
    summaries: dict[tuple[str, int, str], dict[str, float]] = {}
    for seed in protocol["config"]["seeds"]:
        for arm in ARMS:
            old_folder, new_folder = (root / f"seed_{seed}" / arm for root in (baseline, output))
            original = json.loads((old_folder / "complete.json").read_text())
            current = json.loads((new_folder / "complete.json").read_text())
            _verify_pair(old_folder, new_folder, original, current)
            for condition, folder, result in (("constant", old_folder, original), ("expert_lr", new_folder, current)):
                metrics = run_metrics(result)
                evaluations = read_rows(folder / "evaluation.jsonl")
                metrics["time_mean_mse_pass_1"] = _time_mean(evaluations, result["steps_per_epoch"])
                metrics["time_mean_mse_full"] = _time_mean(evaluations, result["steps"])
                summaries[(condition, int(seed), arm)] = metrics
    keys = set.intersection(*(set(values) for values in summaries.values()))
    effects: dict[str, Any] = {}
    condition_stats: dict[str, Any] = {}
    for arm in ARMS:
        effects[arm] = {}
        condition_stats[arm] = {}
        for condition in ("constant", "expert_lr"):
            condition_stats[arm][condition] = {
                key: _interval([summaries[(condition, int(seed), arm)][key] for seed in protocol["config"]["seeds"]])
                for key in sorted(keys)
            }
        for key in sorted(keys):
            differences = [
                summaries[("expert_lr", int(seed), arm)][key] - summaries[("constant", int(seed), arm)][key]
                for seed in protocol["config"]["seeds"]
            ]
            effects[arm][key] = {**_interval(differences), "seed_differences": differences}
    result = {
        "experiment": protocol["experiment"],
        "baseline_directory": str(baseline),
        "verification": {**verification, "matched_every_cross_experiment_initialization_order_view_and_mask": True},
        "learning_rate_audit": learning_rate_audit,
        "all_saved_initial_tensors_match_baseline_exactly": displacements[
            "all_saved_initial_tensors_match_baseline_exactly"
        ],
        "conditions": condition_stats,
        "expert_lr_minus_constant": effects,
        "replicate_unit": "paired training seed",
    }
    write_json(output / "baseline_comparison.json", result)
    import matplotlib.pyplot as plt  # noqa: PLC0415

    for name, figure in (
        ("baseline_learning_curves", plot_baseline_curves(output, baseline)),
        (
            "expert_learning_rates",
            plot_expert_rates(
                output, int(protocol["config"]["seeds"][0]), max(0, protocol["config"]["encoder"]["depth"] - 2)
            ),
        ),
    ):
        figure.savefig(output / "analysis" / f"{name}.png", dpi=150)
        plt.close(figure)
    return result


def plot_baseline_curves(output: Path, baseline: Path) -> Figure:
    """Overlay the old and new experiments, retaining paired data exposure and update axes."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    protocol = json.loads((output / "protocol.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
    for condition, root, style in (("constant LR", baseline, "--"), ("expert LR", output, "-")):
        for arm in ARMS:
            traces = [
                read_rows(root / f"seed_{seed}" / arm / "evaluation.jsonl") for seed in protocol["config"]["seeds"]
            ]
            for ax, key in zip(axes, ("heldout_mse", "rankme"), strict=True):
                values = [{row["step"]: row[key] for row in trace} for trace in traces]
                steps = sorted(set.intersection(*(set(row) for row in values)))
                matrix = np.array([[row[step] for step in steps] for row in values])
                mean = matrix.mean(0)
                sem = matrix.std(0, ddof=1) / np.sqrt(len(matrix)) if len(matrix) > 1 else np.zeros_like(mean)
                ax.plot(steps, mean, color=COLORS[arm], linestyle=style, label=f"{arm}: {condition}")
                ax.fill_between(steps, mean - sem, mean + sem, color=COLORS[arm], alpha=0.08)
                ax.set(xlabel="Gradient update", title=key)
    axes[0].legend(fontsize=8)
    fig.suptitle("Matched seeds and exposures · mean ±1 SEM")
    return fig


def plot_expert_rates(output: Path, seed: int, layer: int) -> Figure:
    """Show probability mass, applied LR, and actual parameter movement on shared scales."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(3, 2, figsize=(15, 9), constrained_layout=True, sharex=True, sharey="row")
    colors = [plt.get_cmap("tab10")(index) for index in range(4)]
    for column, arm in enumerate(ARMS):
        rows = read_rows(output / f"seed_{seed}" / arm / "training.jsonl")
        steps = np.array([row["step"] for row in rows])
        for row_index, (key, label) in enumerate(
            (
                ("mean_probabilities", "Mean probability"),
                ("expert_lr", "Applied learning rate"),
                ("expert_update_norm", "Actual parameter-step L2"),
            )
        ):
            values = np.array([row[f"l{layer}_{key}"] for row in rows])
            for expert, color in enumerate(colors):
                axes[row_index, column].plot(
                    steps,
                    np.maximum(values[:, expert], 1e-16) if row_index == 2 else values[:, expert],
                    color=color,
                    lw=0.7,
                    label=f"Expert {expert}",
                )
            axes[row_index, column].set(ylabel=label, xlabel="Gradient update")
            if row_index == 2:
                axes[row_index, column].set_yscale("log")
        axes[0, column].set_title(f"{arm} · seed {seed} · layer {layer}")
    axes[0, 0].legend(ncols=4, fontsize=8)
    fig.suptitle("Expert-specific learning rates · AdamW moments and counters update normally")
    return fig
