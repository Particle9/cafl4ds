"""Paired reconstruction and forgetting analysis for M6.3.0."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from cafl4ds.cluster_routing_analysis import _interval, read_rows
from cafl4ds.day_night_experiment import ARMS, PHASES, TRUNK_ARMS
from cafl4ds.routing_experiment import state_hash, tensor_hash, write_json

if TYPE_CHECKING:
    from matplotlib.figure import Figure

LABELS = dict(
    zip(
        ARMS,
        (
            "V-MoE top-4",
            "V-MoE top-1",
            "GMM MLP",
            "GMM full trunks",
            "GMM split after 2",
            "GMM full trunks + private decoders",
        ),
        strict=True,
    )
)
COLORS = dict(zip(ARMS, ("#2878b5", "#56a5bd", "#d07b28", "#499957", "#9160a5", "#b34052"), strict=True))


LABELS["gmm_all_layers"] = "GMM MoE in all 4 blocks"
COLORS["gmm_all_layers"] = "#674328"


def read_comparison(output: Path) -> dict[str, Any]:
    """Load the original comparison and the four-layer ablation when available."""
    comparison: dict[str, Any] = json.loads((output / "comparison.json").read_text())
    extra = output / "gmm_all_layers" / "comparison.json"
    if extra.is_file():
        ablation = json.loads(extra.read_text())
        comparison["runs"].extend(ablation["runs"])
        comparison["conditions"].update(ablation["conditions"])
        comparison["paired_forgetting_differences"].extend(ablation["paired_forgetting_differences"])
    return comparison


def run_directory(output: Path, run: dict[str, Any]) -> Path:
    """Resolve the directory recorded for a run."""
    directory: str = run.get("directory", f"seed_{run['seed']}/{run['arm']}")
    return output / directory


def _check_tensors(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], names: list[str]) -> None:
    if not all(torch.equal(left[name], right[name]) for name in names):
        raise ValueError("Matched model components start from different initial tensors.")


def _check_initialization(states: dict[str, dict[str, torch.Tensor]], depth: int) -> None:
    top4, top1, gmm, full, split, separate = (states[arm] for arm in ARMS)
    _check_tensors(top4, top1, list(top4))
    _check_tensors(top4, gmm, list(gmm))
    _check_tensors(full, separate, list(full))
    _check_tensors(gmm, full, [key for key in full if key.startswith("decoder.")])
    _check_tensors(gmm, split, [key for key in split if not key.startswith("branches.")])
    for expert in range(4):
        trunk = "encoder" if expert == 0 else f"branches.{expert - 1}"
        for key, value in split.items():
            prefix = f"branches.{expert}."
            if key.startswith(prefix):
                layer, parameter = key.removeprefix(prefix).split(".", 1)
                component = "norm" if int(layer) == depth - 2 else f"blocks.{int(layer) + 2}"
                if not torch.equal(value, full[f"{trunk}.{component}.{parameter}"]):
                    raise ValueError("A private suffix differs from its matching full-trunk initialization.")


def _audit_stream(rows: list[dict[str, Any]], protocol: dict[str, Any], seed: int, boundary: int) -> None:
    expected = protocol["phase_orders"][str(seed)]
    batch_size = protocol["config"]["batch_size"]
    day_steps = math.ceil(expected[PHASES[0]]["images"] / batch_size)
    if boundary != day_steps or [row["step"] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("The update count or phase boundary differs from the frozen schedule.")
    for phase, subset in zip(PHASES, (rows[:boundary], rows[boundary:]), strict=True):
        ids = torch.tensor([sample for row in subset for sample in row["sample_ids"]])
        batch_lengths = [len(row["sample_ids"]) for row in subset]
        if (
            not all(row["phase"] == phase for row in subset)
            or len(ids) != expected[phase]["images"]
            or tensor_hash(ids) != expected[phase]["sha256"]
            or batch_lengths != [len(batch) for batch in ids.split(batch_size)]
        ):
            raise ValueError(f"The trace differs from the frozen {phase} pass.")


def _audit_gmm(
    rows: list[dict[str, Any]],
    folder: Path,
    protocol: dict[str, Any],
    probabilities: torch.Tensor,
    arm: str,
) -> int:
    batch_size = protocol["config"]["batch_size"]
    depth = protocol["config"]["encoder"]["depth"]
    moe_blocks = protocol["config"]["encoder"]["moe_blocks"]
    layers = [0] if arm in TRUNK_ARMS else (list(range(max(0, depth - 2), depth)) if moe_blocks is None else moe_blocks)
    base_lr = protocol["config"]["optim"]["lr"]
    for row in rows:
        p = probabilities[row["sample_ids"]]
        q = (p / p.sum(dim=1, keepdim=True)).mean(dim=0).numpy()
        if row["auxiliary"] != 0 or len(p) > batch_size:
            raise ValueError("Unexpected loss or batch size in a fixed GMM arm.")
        if arm != "gmm_full_private_decoder" and row["shared_and_router_lr"] != base_lr:
            raise ValueError("Shared parameters must retain the base AdamW learning rate.")
        for layer in layers:
            prefix = f"l{layer}"
            np.testing.assert_allclose(row[f"{prefix}_mean_probabilities"], q, rtol=0, atol=1e-7)
            scales = np.minimum(1, 4 * np.asarray(row[f"{prefix}_mean_probabilities"]))
            np.testing.assert_allclose(row[f"{prefix}_expert_lr_scale"], scales, rtol=0, atol=1e-12)
            np.testing.assert_allclose(row[f"{prefix}_expert_lr"], base_lr * scales, rtol=0, atol=1e-12)
    optimizer = torch.load(folder / "latest.pt", map_location="cpu", weights_only=False)["optimizer"]
    for group in optimizer["param_groups"]:
        for parameter in group["params"]:
            if int(optimizer["state"][parameter]["step"]) != len(rows):
                raise ValueError("AdamW counters must advance continuously through the phase change.")
    return len(rows) * len(layers)


def _audit_seed(
    output: Path,
    protocol: dict[str, Any],
    probabilities: torch.Tensor,
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    states, runs = {}, []
    reference = None
    rate_vectors = 0
    for arm in ARMS:
        folder = output / f"seed_{seed}" / arm
        complete = json.loads((folder / "complete.json").read_text())
        if complete["protocol_hash"] != protocol["protocol_hash"]:
            raise ValueError("A completed arm belongs to a different protocol.")
        rows = read_rows(folder / "training.jsonl")
        _audit_stream(rows, protocol, seed, complete["boundary_step"])
        signature = [(row["views_hash"], row["mask_noise_hash"]) for row in rows]
        if reference is not None and signature != reference:
            raise ValueError("Paired training views or masks differ.")
        reference = signature
        if arm.startswith("gmm"):
            rate_vectors += _audit_gmm(rows, folder, protocol, probabilities, arm)
        initial = torch.load(folder / "initial.pt", map_location="cpu", weights_only=True)["model"]
        if state_hash(initial) != complete["initial_hash"]:
            raise ValueError("Saved initialization differs from the recorded hash.")
        states[arm] = initial
        evaluations = read_rows(folder / "evaluation.jsonl")
        steps = [row["step"] for row in evaluations]
        boundary = complete["boundary_step"]
        if steps != sorted(set(steps)) or not {0, boundary, boundary + 1, len(rows)}.issubset(steps):
            raise ValueError("Missing or duplicate phase-boundary evaluation.")
        if complete["steps"] != len(rows) or complete["pre_night"] != evaluations[steps.index(boundary)]:
            raise ValueError("Completed result does not match the recorded trace.")
        if complete["final"] != evaluations[-1] or steps[-1] != len(rows):
            raise ValueError("Endpoint does not match the final evaluation.")
        runs.append(complete)
    _check_initialization(states, protocol["config"]["encoder"]["depth"])
    return runs, rate_vectors


def analyze(output: Path, probabilities: torch.Tensor) -> dict[str, Any]:
    """Audit all traces and summarize paired seed differences from the top-4 control."""
    protocol = json.loads((output / "protocol.json").read_text())
    audited = [_audit_seed(output, protocol, probabilities, seed) for seed in protocol["config"]["seeds"]]
    runs = [row for seed_runs, _ in audited for row in seed_runs]
    metrics = {
        "day_dusk_pre_night": [row["pre_night"]["day_dusk_mse"] for row in runs],
        "day_dusk_final": [row["final"]["day_dusk_mse"] for row in runs],
        "night_final": [row["final"]["night_mse"] for row in runs],
        "forgetting_mse": [row["forgetting_mse"] for row in runs],
        "peak_forgetting_mse": [row["peak_forgetting_mse"] for row in runs],
    }
    by_arm = {
        arm: {
            name: [value for row, value in zip(runs, values, strict=True) if row["arm"] == arm]
            for name, values in metrics.items()
        }
        for arm in ARMS
    }
    stats = {
        arm: {name: _interval(values) for name, values in per_metric.items()} for arm, per_metric in by_arm.items()
    }
    pairs = [
        *((arm, "vmoe_top4") for arm in ARMS[1:]),
        *((arm, "gmm_capped") for arm in TRUNK_ARMS),
        ("gmm_split", "gmm_full"),
        ("gmm_full_private_decoder", "gmm_full"),
    ]
    effects = []
    for arm, reference in pairs:
        differences = [
            value - baseline
            for value, baseline in zip(by_arm[arm]["forgetting_mse"], by_arm[reference]["forgetting_mse"], strict=True)
        ]
        effects.append({"arm": arm, "reference": reference, **_interval(differences), "seed_differences": differences})
    result = {
        "experiment": protocol["experiment"],
        "runs": runs,
        "conditions": stats,
        "paired_forgetting_differences": effects,
        "audit": {
            "matched_images_views_masks_and_compatible_initial_tensors": True,
            "fixed_gmm_weights_capped_rates_and_continuous_adam_counters": True,
            "gradient_updates": sum(row["steps"] for row in runs),
            "gmm_rate_vectors": sum(vectors for _, vectors in audited),
        },
    }
    write_json(output / "comparison.json", result)
    import matplotlib.pyplot as plt  # noqa: PLC0415

    figure = plot_curves(output, "day_dusk")
    figure.savefig(output / "reconstruction.png", dpi=150)
    plt.close(figure)
    return result


def plot_curves(output: Path, subset: str, seed: int | None = None) -> Figure:
    """Show the selected held-out subset, its night-phase change, and night validation MSE."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    comparison = read_comparison(output)
    boundary = comparison["runs"][0]["boundary_step"]
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for arm in comparison["conditions"]:
        rows = [
            read_rows(run_directory(output, run) / "evaluation.jsonl")
            for run in comparison["runs"]
            if run["arm"] == arm and (seed is None or run["seed"] == seed)
        ]
        steps = np.array([row["step"] for row in rows[0]])
        selected = np.array([[row[f"{subset}_mse"] for row in trace] for trace in rows])
        night = np.array([[row["night_mse"] for row in trace] for trace in rows])
        second = steps >= boundary
        change = selected[:, second] - selected[:, steps == boundary]
        for ax, x, values in zip(
            axes, (steps, steps[second] - boundary, steps), (selected, change, night), strict=True
        ):
            mean = values.mean(axis=0)
            sem = values.std(axis=0, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.zeros_like(mean)
            ax.plot(x, mean, label=LABELS[arm], color=COLORS[arm])
            ax.fill_between(x, mean - sem, mean + sem, color=COLORS[arm], alpha=0.12)
    for ax in (axes[0], axes[2]):
        ax.axvline(boundary, ls="--", color="0.4", lw=1)
        ax.set_xlabel("Optimizer updates")
    axes[1].axhline(0, color="0.5", lw=1)
    axes[1].set_xlabel("Night optimizer updates")
    name = {"day_dusk": "Day + dusk", "day": "Daytime", "dusk": "Dawn/dusk"}[subset]
    axes[0].set(title=f"{name} validation", ylabel="Masked patch MSE")
    axes[1].set(title=f"{name} change since phase boundary", ylabel="MSE difference")
    axes[2].set(title="Night validation", ylabel="Masked patch MSE")
    axes[2].legend(fontsize=8)
    figure.suptitle("Paired seed means and ±1 SEM" if seed is None else f"Seed {seed}")
    return figure
