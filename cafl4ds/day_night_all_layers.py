"""M6.3.1: replace the two early dense MLPs with GMM-routed experts."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from cafl4ds import day_night_experiment as experiment
from cafl4ds.cluster_routing_analysis import _interval, read_rows
from cafl4ds.day_night_analysis import _audit_gmm, _audit_stream
from cafl4ds.day_night_drift import _run as measure_drift
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.routing_experiment import digest_file, write_json
from cafl4ds.ssl.factory import build_mae
from cafl4ds.ssl.mae import MAE

ARM = "gmm_all_layers"
DIRECTORY = "gmm_all_layers"
CONTROL = "gmm_capped"


def _reference_key(key: str) -> str:
    return re.sub(r"^(encoder\.blocks\.[01]\.mlp)\.experts\.\d+\.", r"\1.", key)


def build_matched_mae(encoder: TinyViTEncoder, init_checkpoint: str, **kwargs: Any) -> MAE:  # noqa: ANN401
    """Load the paired model and initialize early experts as copies of its dense MLPs."""
    method = build_mae(encoder=encoder, **kwargs)
    reference = torch.load(init_checkpoint, map_location="cpu", weights_only=True)["model"]
    matched = {
        key: reference[_reference_key(key)] if ".router." not in key else value
        for key, value in method.state_dict().items()
    }
    method.load_state_dict(matched)
    return method


def _job(base: Path, output: Path, seed: int, protocol: dict[str, Any]) -> dict[str, Any]:
    worker = experiment._worker
    if worker is None:
        raise RuntimeError("Experiment data was not initialized.")
    original = base / f"seed_{seed}" / CONTROL
    initial = original / "initial.pt"
    if digest_file(initial) != protocol["initial_checkpoints"][str(seed)]:
        raise ValueError(f"Original initialization changed for seed {seed}.")
    cfg = OmegaConf.merge(
        worker.cfg,
        {"ssl": {"_target_": "cafl4ds.day_night_all_layers.build_matched_mae", "init_checkpoint": str(initial)}},
    )
    folder = output / f"seed_{seed}" / CONTROL
    result = experiment.run_arm(cfg, worker.data, worker.panel, seed, CONTROL, folder, protocol["protocol_hash"])
    reference = torch.load(initial, map_location="cpu", weights_only=True)["model"]
    matched = torch.load(folder / "initial.pt", map_location="cpu", weights_only=True)["model"]
    if any(not torch.equal(value, reference[_reference_key(key)]) for key, value in matched.items()):
        raise ValueError(f"Initial tensors did not match for seed {seed}.")
    rows = read_rows(folder / "training.jsonl")
    _audit_stream(rows, protocol, seed, result["boundary_step"])
    paired_fields = ("step", "phase", "sample_ids", "views_hash", "mask_noise_hash")
    for new, old in zip(rows, read_rows(original / "training.jsonl"), strict=True):
        if any(new[key] != old[key] for key in paired_fields):
            raise ValueError(f"Training stream differs from the paired control for seed {seed}.")
    before = read_rows(original / "evaluation.jsonl")
    after = read_rows(folder / "evaluation.jsonl")
    if [row["step"] for row in before] != [row["step"] for row in after]:
        raise ValueError(f"Evaluation steps differ for seed {seed}.")
    for key, value in before[0].items():
        np.testing.assert_allclose(after[0][key], value, atol=1e-6, rtol=0)
    _audit_gmm(rows, folder, protocol, worker.data.probabilities, CONTROL)
    measure_drift(output, seed, CONTROL, False, data=worker.data)
    return {**result, "arm": ARM, "directory": str(folder.relative_to(base))}


def _summarize(base: Path, runs: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        "day_dusk_pre_night": [run["pre_night"]["day_dusk_mse"] for run in runs],
        "day_dusk_final": [run["final"]["day_dusk_mse"] for run in runs],
        "night_final": [run["final"]["night_mse"] for run in runs],
        "forgetting_mse": [run["forgetting_mse"] for run in runs],
        "peak_forgetting_mse": [run["peak_forgetting_mse"] for run in runs],
    }
    comparison = json.loads((base / "comparison.json").read_text())
    effects = []
    for arm in ("vmoe_top4", CONTROL, "gmm_full", "gmm_split", "gmm_full_private_decoder"):
        original = {run["seed"]: run["forgetting_mse"] for run in comparison["runs"] if run["arm"] == arm}
        differences = [run["forgetting_mse"] - original[run["seed"]] for run in runs]
        effects.append({"arm": ARM, "reference": arm, **_interval(differences), "seed_differences": differences})
    return {
        "runs": runs,
        "conditions": {ARM: {key: _interval(values) for key, values in metrics.items()}},
        "paired_forgetting_differences": effects,
    }


def run_experiment(base: Path) -> None:
    """Train paired seeds and recover drift at all original evaluation steps."""
    base = base.resolve()
    output = base / DIRECTORY
    output.mkdir(exist_ok=True)
    protocol = json.loads((base / "protocol.json").read_text())
    protocol.pop("protocol_hash")
    cfg = protocol["config"]
    seeds = cfg["seeds"]
    if cfg["encoder"]["depth"] != 4 or cfg["encoder"]["moe_blocks"] not in (None, [2, 3]):
        raise ValueError("The reference must have four blocks with MoE in the last two.")
    cfg["encoder"]["moe_blocks"] = [0, 1, 2, 3]
    cfg["experiment_id"] = "M6.3.1"
    protocol["experiment"] = "M6.3.1"
    protocol["arms"] = [CONTROL]
    protocol["initial_checkpoints"] = {
        str(seed): digest_file(base / f"seed_{seed}" / CONTROL / "initial.pt") for seed in seeds
    }
    protocol["source_hashes"][Path(__file__).name] = digest_file(Path(__file__))
    protocol["protocol_hash"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    path = output / "protocol.json"
    if path.is_file() and json.loads(path.read_text()) != protocol:
        raise ValueError("Refusing to resume with a different protocol.")
    write_json(path, protocol)
    with ProcessPoolExecutor(
        max_workers=int(cfg["workers"]),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=experiment._initialize_worker,
        initargs=(cfg, output),
    ) as executor:
        futures = [executor.submit(_job, base, output, int(seed), protocol) for seed in seeds]
        runs = [future.result() for future in as_completed(futures)]
    runs.sort(key=lambda run: run["seed"])
    write_json(output / "comparison.json", _summarize(base, runs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, nargs="?", default=Path("outputs/m6_3_0"))
    run_experiment(parser.parse_args().reference)
