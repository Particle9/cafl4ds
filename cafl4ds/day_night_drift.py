"""Recover representation metrics at the M6.3.0 evaluation steps."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from cafl4ds.cluster_routing_analysis import read_rows
from cafl4ds.cluster_routing_experiment import ExperimentData, load_data
from cafl4ds.day_night_analysis import COLORS, LABELS, read_comparison, run_directory
from cafl4ds.day_night_experiment import (
    PHASES,
    _conditioning,
    _time_labels,
    _update,
    build_arm,
    optimizer_groups,
)
from cafl4ds.measurements import cosine_drift, linear_cka
from cafl4ds.routing_experiment import digest_file, paired_views, tensor_hash, write_json
from cafl4ds.ssl.mae import MAE

if TYPE_CHECKING:
    from matplotlib.figure import Figure

SUBSET_LABELS = {"day_dusk": "Day + dusk", "day": "Daytime", "dusk": "Dawn/dusk", "night": "Night"}
PROBES = ("initial", "pre_night")
METRICS = ("cka", "cosine_drift")
_data: ExperimentData | None = None


def _load_inputs(output: Path) -> None:
    global _data  # noqa: PLW0603
    protocol = json.loads((output / "protocol.json").read_text())
    cfg = OmegaConf.create(protocol["config"])
    torch.set_num_threads(int(cfg.threads))
    if str(torch.__version__) != str(protocol["torch"]):
        raise ValueError("Replay requires the original Torch version.")
    for name, digest in protocol["source_hashes"].items():
        if digest_file(Path(__file__).parent / name) != digest:
            raise ValueError(f"Replay requires the frozen training source: {name}")
    data = load_data(cfg)
    if (
        data.corpus.fingerprint != protocol["data"]
        or json.loads(json.dumps(data.provenance)) != protocol["conditioning"]
    ):
        raise ValueError("The training cache or validation panel differs from the original run.")
    _data = data


@torch.no_grad()  # type: ignore[misc]
def _embeddings(method: MAE, data: ExperimentData, batch_size: int) -> torch.Tensor:
    method.eval()
    panel = data.regime_panel
    features = []
    for images, probabilities in zip(
        panel.images.split(batch_size), panel.probabilities.split(batch_size), strict=True
    ):
        with _conditioning(method, probabilities):
            features.append(method.encode(images).detach().cpu())
    return torch.cat(features)


def _subsets(data: ExperimentData) -> dict[str, torch.Tensor]:
    labels = _time_labels(data.regime_panel.regimes, data.corpus.regime_names, bool(data.provenance.get("synthetic")))
    return {"day_dusk": labels < 2, "day": labels == 0, "dusk": labels == 1, "night": labels == 2}


def _measure(
    step: int,
    embeddings: torch.Tensor,
    references: dict[str, torch.Tensor],
    subsets: dict[str, torch.Tensor],
) -> dict[str, Any]:
    scores = {}
    for name, selected in subsets.items():
        if int(selected.sum()) < 2:
            continue
        current = embeddings[selected]
        scores[name] = {
            key: {
                "cka": linear_cka(reference[selected], current),
                "cosine_drift": max(0.0, cosine_drift(reference[selected], current)),
            }
            for key, reference in references.items()
        }
    return {"step": step, "subsets": scores}


def _run(output: Path, seed: int, arm: str, checkpoints_only: bool, *, data: ExperimentData | None = None) -> None:
    data = _data if data is None else data
    if data is None:
        raise RuntimeError("Replay inputs were not loaded.")
    folder = output / f"seed_{seed}" / arm
    target = folder / ("drift_checkpoints.json" if checkpoints_only else "drift.json")
    if target.exists():
        return
    complete = json.loads((folder / "complete.json").read_text())
    protocol = json.loads((output / "protocol.json").read_text())
    cfg = OmegaConf.create(protocol["config"])
    boundary, end = complete["boundary_step"], complete["steps"]
    method = build_arm(cfg, seed, arm)
    states = {
        step: torch.load(folder / name, map_location="cpu", weights_only=True)["model"]
        for step, name in ((0, "initial.pt"), (boundary, "pre_night.pt"), (end, "latest.pt"))
    }
    subsets = _subsets(data)
    references: dict[str, torch.Tensor] = {}
    rows = []
    if checkpoints_only:
        for step, state in states.items():
            method.load_state_dict(state)
            features = _embeddings(method, data, int(cfg.batch_size))
            if step == 0:
                references["initial"] = features
            if step == boundary:
                references["pre_night"] = features
            rows.append(_measure(step, features, references, subsets))
    else:
        method.load_state_dict(states[0])
        features = _embeddings(method, data, int(cfg.batch_size))
        references["initial"] = features
        rows = [_measure(0, features, references, subsets)]
        rows.extend(_replay(folder, method, data, cfg, seed, arm, boundary, states, references, subsets))
    write_json(target, {"seed": seed, "arm": arm, "protocol_hash": protocol["protocol_hash"], "rows": rows})
    print(f"{arm} seed {seed}: {len(rows)} measured steps", flush=True)


def _replay(
    folder: Path,
    method: MAE,
    data: ExperimentData,
    cfg: DictConfig,
    seed: int,
    arm: str,
    boundary: int,
    states: dict[int, dict[str, torch.Tensor]],
    references: dict[str, torch.Tensor],
    subsets: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    optimizer = instantiate(cfg.optim, _partial_=True)(params=optimizer_groups(method, arm, float(cfg.optim.lr)))
    eval_steps = {row["step"] for row in read_rows(folder / "evaluation.jsonl")}
    rows = []
    for original in read_rows(folder / "training.jsonl"):
        step = original["step"]
        ids = torch.tensor(original["sample_ids"])
        phase = PHASES.index(original["phase"])
        views, noise = paired_views(method, data.corpus.images[ids], ids, seed, phase)
        if tensor_hash(views) != original["views_hash"] or tensor_hash(noise) != original["mask_noise_hash"]:
            raise ValueError(f"Replay input mismatch in {arm}, seed {seed}, step {step}.")
        signals = _update(method, optimizer, views, noise, data.probabilities[ids], cfg, seed, step - 1, arm)
        if any(value != original[key] for key, value in signals.items()):
            raise ValueError(f"Replay training trace mismatch in {arm}, seed {seed}, step {step}.")
        if step in states and any(
            not torch.equal(value, states[step][key]) for key, value in method.state_dict().items()
        ):
            raise ValueError(f"Replay checkpoint mismatch in {arm}, seed {seed}, step {step}.")
        if step in eval_steps:
            features = _embeddings(method, data, int(cfg.batch_size))
            if step == boundary:
                references["pre_night"] = features
            rows.append(_measure(step, features, references, subsets))
    if {0, *(row["step"] for row in rows)} != eval_steps:
        raise ValueError("Replay did not cover every evaluation step.")
    return rows


def main() -> None:
    """Measure saved checkpoints or replay the original runs at the original cadence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoints-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    protocol = json.loads((output / "protocol.json").read_text())
    jobs = [(seed, arm) for arm in reversed(protocol["arms"]) for seed in protocol["config"]["seeds"]]
    with ProcessPoolExecutor(
        max_workers=int(protocol["config"]["workers"]),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_load_inputs,
        initargs=(output,),
    ) as executor:
        futures = [executor.submit(_run, output, seed, arm, args.checkpoints_only) for seed, arm in jobs]
        for future in as_completed(futures):
            future.result()


def plot_representation_drift(output: Path, subset: str, seed: int | None = None) -> Figure:
    """Plot CKA similarity and cosine drift against initialization and the switch."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    comparison = read_comparison(output)
    runs = [run for run in comparison["runs"] if seed is None or run["seed"] == seed]
    folders = [(run["arm"], run_directory(output, run)) for run in runs]
    full = all((folder / "drift.json").is_file() for _, folder in folders)
    filename = "drift.json" if full else "drift_checkpoints.json"
    boundary = runs[0]["boundary_step"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col", constrained_layout=True)
    for arm in comparison["conditions"]:
        label = LABELS[arm]
        records = [json.loads((folder / filename).read_text())["rows"] for key, folder in folders if key == arm]
        for column, reference in enumerate(PROBES):
            reference_step = 0 if reference == "initial" else boundary
            steps = [row["step"] for row in records[0] if row["step"] >= reference_step]
            for index, metric in enumerate(METRICS):
                values = np.array(
                    [
                        [row["subsets"][subset][reference][metric] for row in trace if row["step"] >= reference_step]
                        for trace in records
                    ]
                )
                mean = values.mean(axis=0)
                ax = axes[index, column]
                ax.plot(
                    steps,
                    mean,
                    color=COLORS[arm],
                    label=label,
                    marker=None if full else "o",
                    linestyle="-" if full else "none",
                )
                if full and len(values) > 1:
                    sem = values.std(axis=0, ddof=1) / np.sqrt(len(values))
                    ax.fill_between(steps, mean - sem, mean + sem, color=COLORS[arm], alpha=0.12)
    for ax in axes.flat:
        ax.axvline(boundary, color="0.4", linestyle="--", linewidth=0.8, zorder=0)
    axes[0, 0].set_title("Relative to initialization")
    axes[0, 1].set_title("Relative to the pre-night checkpoint")
    for row, label in zip(axes, ("Linear CKA", "Cosine drift (1 − mean cosine)"), strict=True):
        row[0].set_ylabel(label)
    for ax in axes[-1, :]:
        ax.set_xlabel("Optimizer updates")
    axes[0, 1].legend(fontsize=8)
    seeds = "Seed means" if seed is None else f"Seed {seed}"
    coverage = "all evaluation steps" if full else "saved checkpoints"
    fig.suptitle(f"{SUBSET_LABELS[subset]} validation · {seeds} · {coverage}")
    return fig


if __name__ == "__main__":
    main()
