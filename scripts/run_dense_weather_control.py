"""Train a paired dense MAE control and evaluate it on both saved M6.4.0 probe panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms

from cafl4ds import day_night_experiment as experiment
from cafl4ds import weather_probe
from cafl4ds.cluster_routing_analysis import _interval, read_rows
from cafl4ds.day_night_analysis import _audit_stream
from cafl4ds.routing_experiment import CachedRoutingCorpus, digest_file, write_json
from cafl4ds.weather_probe import (
    DENSE_ARM,
    DENSE_REFERENCE_ARM,
    DOMAINS,
    TRANSFER_METRICS,
    WEATHER,
    build_dense_control,
    score_transfer,
    summarize_transfer,
    transfer_gap,
)


def _train_seed(reference: Path, output: Path, protocol: dict[str, Any], seed: int) -> dict[str, Any]:
    worker = experiment._worker
    if worker is None:
        raise RuntimeError("Training inputs were not initialized.")
    initial = reference / f"seed_{seed}" / DENSE_REFERENCE_ARM / "initial.pt"
    if digest_file(initial) != protocol["paired_initial_checkpoints"][str(seed)]:
        raise ValueError(f"Paired initialization changed for seed {seed}.")
    cfg = OmegaConf.create(protocol["config"])
    directory = output / f"seed_{seed}" / DENSE_ARM
    # Inject only the model factory in this worker. The frozen M6.3 runner still owns batching,
    # optimizer state, augmentation, validation and checkpoints, and remains replayable as-is.
    with patch.object(experiment, "build_arm", partial(build_dense_control, initial=initial)):
        result = experiment.run_arm(
            cfg, worker.data, worker.panel, seed, DENSE_ARM, directory, protocol["protocol_hash"]
        )
    trace = read_rows(directory / "training.jsonl")
    _audit_stream(trace, protocol, seed, result["boundary_step"])
    reference_trace = read_rows(initial.parent / "training.jsonl")
    paired = ("step", "phase", "sample_ids", "views_hash", "mask_noise_hash")
    for actual, expected in zip(trace, reference_trace, strict=True):
        if any(actual[key] != expected[key] for key in paired) or actual["auxiliary"] != 0.0:
            raise ValueError(f"Unmatched training inputs or nonzero auxiliary loss for seed {seed}.")
    return result


def _load_probe_images(cfg: DictConfig, panels: dict[str, list[dict[str, str]]]) -> tuple[torch.Tensor, dict[str, int]]:
    # The workers verified the full cache against the reference before training.
    corpus = CachedRoutingCorpus(cfg.source.directory, verify_hashes=False)
    train_index = {name: i for i, name in enumerate(corpus.names)}
    val_dir = Path(corpus.metadata["bdd_root"]) / "images/100k/val"
    transform = transforms.Compose([transforms.Resize((cfg.img_size, cfg.img_size)), transforms.ToTensor()])
    images: list[torch.Tensor] = []
    indices = {}
    for name, panel in panels.items():
        for row in panel:
            image_name = row["image"]
            if image_name in indices:
                continue
            indices[image_name] = len(images)
            if name == "train_2000" and row["partition"] == "train":
                images.append(corpus.images[train_index[image_name]])
            else:
                with Image.open(val_dir / image_name) as image:
                    images.append(transform(image.convert("RGB")))
    return torch.stack(images), indices


def _evaluate_probes(
    experiment_output: Path, reference: Path, weather: Path, training: dict[str, Any], runs: list[dict[str, Any]]
) -> None:
    cfg = OmegaConf.create(training["config"])
    torch.set_num_threads(int(cfg.threads))
    protocols = {name: weather / name / "protocol.json" for name in (".", "train_2000")}
    if any(digest_file(path) != training["probe_panel_hashes"][name] for name, path in protocols.items()):
        raise ValueError("A saved weather-probe panel changed during dense training.")
    panels = {name: json.loads(path.read_text())["panel"] for name, path in protocols.items()}
    inputs, indices = _load_probe_images(cfg, panels)

    results: dict[str, list[dict[str, Any]]] = {name: [] for name in panels}
    for run in runs:
        seed = run["seed"]
        initial = reference / f"seed_{seed}" / DENSE_REFERENCE_ARM / "initial.pt"
        method = build_dense_control(cfg, seed, DENSE_ARM, initial=initial)
        checkpoint = experiment_output / f"seed_{seed}" / DENSE_ARM / "latest.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["protocol_hash"] != training["protocol_hash"] or state["step"] != run["steps"]:
            raise ValueError(f"Mismatched final dense checkpoint for seed {seed}.")
        method.load_state_dict(state["model"])
        method.eval()
        features = torch.cat([method.encode(batch).cpu() for batch in inputs.split(cfg.batch_size)]).numpy()
        for name, panel in panels.items():
            labels = np.asarray([WEATHER.index(row["weather"]) for row in panel])
            train, test = (
                {
                    domain: np.asarray(
                        [i for i, row in enumerate(panel) if row["timeofday"] == domain and row["partition"] == part]
                    )
                    for domain in DOMAINS
                }
                for part in ("train", "test")
            )
            scores = score_transfer(features[[indices[row["image"]] for row in panel]], labels, train, test)
            results[name].append(
                {
                    "arm": DENSE_ARM,
                    "seed": seed,
                    "step": state["step"],
                    "sha256": digest_file(checkpoint),
                    "scores": scores,
                    "transfer_gap": {m: transfer_gap(scores, m) for m in TRANSFER_METRICS},
                }
            )
        logger.info("Dense seed {} scored on both weather-probe panels", seed)

    for name, rows in results.items():
        stats = summarize_transfer(rows)
        moe = json.loads((weather / name / "results.json").read_text())
        dense_gap = {row["seed"]: row["transfer_gap"]["rainy_f1"] for row in rows}
        contrasts = {}
        for arm in moe["summary"]:
            moe_gap = {row["seed"]: row["transfer_gap"]["rainy_f1"] for row in moe["runs"] if row["arm"] == arm}
            if set(moe_gap) != set(dense_gap):
                raise ValueError(f"Seeds differ between the dense control and {arm}.")
            contrasts[arm] = _interval([moe_gap[seed] - dense_gap[seed] for seed in sorted(dense_gap)])
        destination = experiment_output / name
        destination.mkdir(exist_ok=True)
        write_json(
            destination / "results.json",
            {
                "panel_protocol_sha256": digest_file(protocols[name]),
                "runs": rows,
                "summary": {DENSE_ARM: stats},
                "paired_f1_gap_vs_dense": contrasts,
            },
        )


def main() -> None:
    """Train five dense checkpoints on the frozen M6.3 stream and fit matched probes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("outputs/m6_3_0"))
    parser.add_argument("--weather", type=Path, default=Path("outputs/m6_4_0"))
    args = parser.parse_args()
    source = json.loads((args.reference / "protocol.json").read_text())
    seeds = source["config"]["seeds"]
    output = args.weather / "dense_control"
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        **source,
        "experiment": "M6.4.0 dense control",
        "arms": [DENSE_ARM],
        "config": OmegaConf.to_container(
            OmegaConf.merge(source["config"], {"encoder": {"moe_blocks": []}, "ssl": {"aux_weight": 0}}), resolve=True
        ),
        "paired_protocol_hash": source["protocol_hash"],
        "probe_panel_hashes": {
            name: digest_file(args.weather / name / "protocol.json") for name in (".", "train_2000")
        },
        "paired_initial_checkpoints": {
            str(seed): digest_file(args.reference / f"seed_{seed}" / DENSE_REFERENCE_ARM / "initial.pt")
            for seed in seeds
        },
        "dense_runner_sha256": digest_file(Path(__file__)),
        "weather_probe_sha256": digest_file(Path(weather_probe.__file__)),
    }
    del protocol["protocol_hash"]
    protocol["protocol_hash"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    path = output / "protocol.json"
    if path.is_file() and json.loads(path.read_text()) != protocol:
        raise ValueError("Refusing to resume a different dense-control protocol.")
    write_json(path, protocol)
    with ProcessPoolExecutor(
        max_workers=source["config"]["workers"],
        mp_context=multiprocessing.get_context("spawn"),
        initializer=experiment._initialize_worker,
        initargs=(source["config"], output),
    ) as pool:
        futures = [pool.submit(_train_seed, args.reference, output, protocol, seed) for seed in seeds]
        runs = [future.result() for future in futures]
    write_json(output / "training.json", {"runs": runs})
    _evaluate_probes(output, args.reference, args.weather, protocol, runs)


if __name__ == "__main__":
    main()
