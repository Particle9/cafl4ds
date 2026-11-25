"""Evaluate one pooled weather probe per final checkpoint for M6.4.1."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cafl4ds import weather_probe
from cafl4ds.cluster_routing_analysis import _interval
from cafl4ds.day_night_all_layers import ARM as ALL_LAYERS_ARM
from cafl4ds.day_night_all_layers import DIRECTORY as ALL_LAYERS_DIR
from cafl4ds.day_night_analysis import read_comparison, run_directory
from cafl4ds.day_night_experiment import ARMS, _conditioning, build_arm
from cafl4ds.routing_experiment import CachedRoutingCorpus, digest_file, write_json
from cafl4ds.weather_probe import DENSE_ARM, DENSE_REFERENCE_ARM, WEATHER, build_dense_control

METRICS = ("accuracy", "balanced_accuracy", "rainy_f1", "roc_auc", "average_precision")
_inputs: tuple[np.ndarray, np.ndarray, np.ndarray, int] | None = None


def _select_panel(root: Path, routes: dict[str, int]) -> list[dict[str, Any]]:
    panel = []
    for split in ("train", "val"):
        paths = sorted((root / "labels/100k" / split).glob("*.json"))
        if len(paths) != sum(name.startswith(f"images/100k/{split}/") for name in routes):
            raise ValueError(f"Expected a BDD label record for each {split} image.")
        for path in paths:
            record = json.loads(path.read_text())
            weather = record["attributes"]["weather"]
            if weather in WEATHER:
                name = record["name"]
                if not Path(name).suffix:
                    name += ".jpg"
                image = f"images/100k/{split}/{name}"
                panel.append({"image": image, "split": split, "weather": weather, "route": routes[image]})
    if len({Path(row["image"]).name for row in panel}) != len(panel):
        raise ValueError("Image IDs must be unique across the train and validation splits.")
    return panel


def _prepare_images(root: Path, reference: dict[str, Any], panel: list[dict[str, Any]], output: Path) -> None:
    cfg = reference["config"]
    corpus = CachedRoutingCorpus(cfg["source"]["directory"])
    if corpus.fingerprint != reference["data"]:
        raise ValueError("Training image cache differs from the M6.3 reference.")
    cache = {name: i for i, name in enumerate(corpus.names)}
    size = cfg["img_size"]
    temporary = output.with_suffix(".tmp.npy")
    packed = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8, shape=(len(panel), 3, size, size))
    for i, row in enumerate(panel):
        name = Path(row["image"]).name
        if row["split"] == "train" and name in cache:
            packed[i] = np.rint(corpus.array[cache[name]] * 255).astype(np.uint8)
        else:
            with Image.open(root / row["image"]) as image:
                resized = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
                packed[i] = np.asarray(resized).transpose(2, 0, 1)
    packed.flush()
    del packed
    temporary.replace(output)


def _initialize(output: Path, probability_path: Path, threads: int) -> None:
    global _inputs  # noqa: PLW0603
    torch.set_num_threads(threads)
    panel = json.loads((output / "panel.json").read_text())
    images = np.load(output / "images.npy", mmap_mode="r", allow_pickle=False)
    probabilities = np.load(probability_path, mmap_mode="r", allow_pickle=False)[[row["route"] for row in panel]]
    labels = np.asarray([WEATHER.index(row["weather"]) for row in panel])
    boundary = sum(row["split"] == "train" for row in panel)
    _inputs = images, probabilities, labels, boundary


@torch.inference_mode()  # type: ignore[misc]
def _encode_checkpoint(job: dict[str, Any], images: np.ndarray, probabilities: np.ndarray, batch: int) -> np.ndarray:
    cfg = OmegaConf.create(job["config"])
    arm, seed = job["arm"], job["seed"]
    if arm == DENSE_ARM:
        method = build_dense_control(cfg, seed, arm, initial=Path(job["initial"]))
    else:
        method = build_arm(cfg, seed, "gmm_capped" if arm == ALL_LAYERS_ARM else arm)
    state = torch.load(job["checkpoint"], weights_only=True, map_location="cpu")
    if state["protocol_hash"] != job["training_protocol"] or state["step"] != job["step"]:
        raise ValueError(f"Checkpoint provenance mismatch for {arm} seed {seed}.")
    method.load_state_dict(state["model"])
    method.eval()
    features = np.empty((len(images), cfg.encoder.embed_dim), dtype=np.float32)
    for start in range(0, len(images), batch):
        stop = min(start + batch, len(images))
        inputs = torch.from_numpy(images[start:stop].copy()).float().div_(255)
        with _conditioning(method, torch.from_numpy(probabilities[start:stop])):
            features[start:stop] = method.encode(inputs).cpu().numpy()
    return features


def _score(job: dict[str, Any], output: Path, protocol_hash: str, batch: int) -> dict[str, Any]:
    if _inputs is None:
        raise RuntimeError("Probe inputs were not initialized.")
    images, probabilities, labels, boundary = _inputs
    arm, seed = job["arm"], job["seed"]
    path = output / "runs" / f"{arm}_{seed}.json"
    checkpoint_hash = digest_file(Path(job["checkpoint"]))
    if path.exists():
        cached: dict[str, Any] = json.loads(path.read_text())
        if cached["protocol_hash"] != protocol_hash or cached["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError(f"Stale result at {path}.")
        return cached
    features = _encode_checkpoint(job, images, probabilities, batch)
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced"))
    probe.fit(features[:boundary], labels[:boundary])
    val_x, val_y = features[boundary:], labels[boundary:]
    logits = probe.decision_function(val_x)
    predictions = probe.predict(val_x)
    result = {
        "arm": arm,
        "seed": seed,
        "step": job["step"],
        "protocol_hash": protocol_hash,
        "checkpoint_sha256": checkpoint_hash,
        "scores": {
            "accuracy": accuracy_score(val_y, predictions),
            "balanced_accuracy": balanced_accuracy_score(val_y, predictions),
            "rainy_f1": f1_score(val_y, predictions, zero_division=0),
            "roc_auc": roc_auc_score(val_y, logits),
            "average_precision": average_precision_score(val_y, logits),
            "confusion_matrix": confusion_matrix(val_y, predictions, labels=[0, 1]).tolist(),
        },
    }
    write_json(path, result)
    return result


def _jobs(reference: Path, dense: Path, original: dict[str, Any]) -> list[dict[str, Any]]:
    full = read_comparison(reference)["runs"]
    dense_runs = json.loads((dense / "training.json").read_text())["runs"]
    protocols = dict.fromkeys(ARMS, original) | {
        ALL_LAYERS_ARM: json.loads((reference / ALL_LAYERS_DIR / "protocol.json").read_text()),
        DENSE_ARM: json.loads((dense / "protocol.json").read_text()),
    }
    runs = {(row["arm"], row["seed"]): row for row in full + dense_runs}
    keys = [(arm, seed) for arm in protocols for seed in original["config"]["seeds"]]
    if set(runs) != set(keys) or len(full) + len(dense_runs) != len(keys):
        raise ValueError("Expected the same five seeds for every architecture.")
    jobs = []
    for arm, seed in keys:
        run = runs[arm, seed]
        location = dense / f"seed_{seed}" / arm if arm == DENSE_ARM else run_directory(reference, run)
        jobs.append(
            {
                "arm": arm,
                "seed": seed,
                "step": run["steps"],
                "checkpoint": str(location / "latest.pt"),
                "initial": str(reference / f"seed_{seed}" / DENSE_REFERENCE_ARM / "initial.pt"),
                "config": protocols[arm]["config"],
                "training_protocol": protocols[arm]["protocol_hash"],
            }
        )
    return jobs


def _prepare(args: argparse.Namespace, original: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    output = args.output
    gmm = Path(original["conditioning"]["gmm_directory"])
    manifest_path, probability_path = gmm / "manifest.json", gmm / "probabilities.npy"
    for key, path in (("gmm_manifest_sha256", manifest_path), ("gmm_probability_sha256", probability_path)):
        if digest_file(path) != original["conditioning"][key]:
            raise ValueError(f"Routing cache differs from M6.3 at {path}.")
    manifest = json.loads(manifest_path.read_text())
    root = args.bdd_root or Path(manifest["bdd_root"])
    panel_path, image_path = output / "panel.json", output / "images.npy"
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        panel = json.loads(panel_path.read_text())
    else:
        routes = {row["path"]: i for i, row in enumerate(manifest["images"])}
        logger.info("Reading official BDD train and validation weather labels")
        panel = _select_panel(root, routes)
        write_json(panel_path, panel)
        logger.info("Preparing {:,} images at {} pixels", len(panel), original["config"]["img_size"])
        _prepare_images(root, original, panel, image_path)
    counts = {
        split: dict(Counter(row["weather"] for row in panel if row["split"] == split)) for split in ("train", "val")
    }
    protocol = {
        "experiment": "M6.4.1",
        "class_counts": counts,
        "panel_sha256": digest_file(panel_path),
        "images_sha256": digest_file(image_path),
        "routing_probabilities_sha256": digest_file(probability_path),
        "runner_sha256": digest_file(Path(__file__)),
        "weather_probe_sha256": digest_file(Path(weather_probe.__file__)),
        "batch_size": args.batch_size,
        "torch_threads": args.threads,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "classifier": {"C": 1.0, "class_weight": "balanced", "max_iter": 1000},
    }
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Output directory already contains a different frozen protocol.")
    write_json(protocol_path, protocol)
    return probability_path, protocol


def main() -> None:
    """Run all frozen checkpoints against the same official train and validation partitions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("outputs/m6_3_0"))
    parser.add_argument("--dense", type=Path, default=Path("outputs/m6_4_0/dense_control"))
    parser.add_argument("--output", type=Path, default=Path("outputs/m6_4_1"))
    parser.add_argument("--bdd-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    original = json.loads((args.reference / "protocol.json").read_text())
    jobs = _jobs(args.reference, args.dense, original)
    (args.output / "runs").mkdir(parents=True, exist_ok=True)
    probability_path, protocol = _prepare(args, original)
    logger.info("Frozen class counts {}", protocol["class_counts"])
    protocol_hash = digest_file(args.output / "protocol.json")
    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize,
        initargs=(args.output, probability_path, args.threads),
    ) as pool:
        futures = [pool.submit(_score, job, args.output, protocol_hash, args.batch_size) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            logger.info(
                "{:02d}/{} {} seed {} F1 {:.3f}",
                len(results),
                len(jobs),
                row["arm"],
                row["seed"],
                row["scores"]["rainy_f1"],
            )
    arm_order = [*ARMS, ALL_LAYERS_ARM, DENSE_ARM]
    results.sort(key=lambda r: (arm_order.index(r["arm"]), r["seed"]))
    summary = {
        arm: {metric: _interval([r["scores"][metric] for r in results if r["arm"] == arm]) for metric in METRICS}
        for arm in arm_order
    }
    write_json(args.output / "results.json", {"runs": results, "summary": summary})


if __name__ == "__main__":
    main()
