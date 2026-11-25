"""Run M6.4.0 day/night weather probes on the final M6.3 checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms

from cafl4ds.day_night_all_layers import ARM as ALL_LAYERS_ARM
from cafl4ds.day_night_all_layers import DIRECTORY as ALL_LAYERS_DIR
from cafl4ds.day_night_analysis import read_comparison, run_directory
from cafl4ds.day_night_experiment import ARMS, _conditioning, build_arm
from cafl4ds.routing_experiment import CachedRoutingCorpus, digest_file, write_json
from cafl4ds.weather_probe import DOMAINS, TRANSFER_METRICS, WEATHER, score_transfer, summarize_transfer, transfer_gap

SPLIT_SEED = 640
TEST_SIZE = 0.30


def _load_panel(
    gmm_dir: Path, bdd_root: Path | None, img_size: int, provenance: dict[str, Any]
) -> tuple[list[str], np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    manifest_path = gmm_dir / "manifest.json"
    probability_path = gmm_dir / "probabilities.npy"
    for name, path in (("manifest", manifest_path), ("probability", probability_path)):
        if digest_file(path) != provenance[f"gmm_{name}_sha256"]:
            raise ValueError(f"{path} differs from the M6.3 routing cache.")
    manifest = json.loads(manifest_path.read_text())
    root = bdd_root or Path(manifest["bdd_root"])
    val_images = root / "images/100k/val"
    rows = {Path(row["path"]).name: index for index, row in enumerate(manifest["images"]) if row["split"] == "val"}
    transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    names, domains, labels, images = [], [], [], []
    for path in sorted((root / "labels/100k/val").glob("*.json")):
        record = json.loads(path.read_text())
        attrs = record["attributes"]
        domain, weather = attrs["timeofday"], attrs["weather"]
        if domain not in DOMAINS or weather not in WEATHER:
            continue
        name = record["name"]
        if not Path(name).suffix:
            name += ".jpg"
        if name not in rows:
            raise ValueError(f"Validation image {name} has no saved GMM posterior.")
        with Image.open(val_images / name) as image:
            images.append(transform(image.convert("RGB")))
        names.append(name)
        domains.append(domain)
        labels.append(WEATHER.index(weather))
    if not names or len(set(names)) != len(names):
        raise ValueError("Expected distinct labeled validation images.")
    probabilities = np.load(probability_path, allow_pickle=False, mmap_mode="r")
    return (
        names,
        np.asarray(domains),
        np.asarray(labels),
        torch.stack(images),
        torch.from_numpy(probabilities[[rows[name] for name in names]].copy()),
    )


def _load_train_probe_panel(
    original: dict[str, Any], gmm_dir: Path, bdd_root: Path | None, test_protocol: Path, count: int
) -> tuple[list[dict[str, str]], torch.Tensor, torch.Tensor]:
    saved = json.loads(test_protocol.read_text())
    provenance = original["conditioning"]
    manifest_path = gmm_dir / "manifest.json"
    probability_path = gmm_dir / "probabilities.npy"
    for key, path in (("gmm_manifest_sha256", manifest_path), ("gmm_probability_sha256", probability_path)):
        if digest_file(path) != provenance[key] or saved[key] != provenance[key]:
            raise ValueError(f"{path} differs from the original probe routing cache.")
    manifest = json.loads(manifest_path.read_text())
    rows = {row["path"]: index for index, row in enumerate(manifest["images"])}
    root = bdd_root or Path(manifest["bdd_root"])
    corpus = CachedRoutingCorpus(original["config"]["source"]["directory"])
    if corpus.fingerprint != original["data"]:
        raise ValueError("Probe training cache differs from the encoder training cache.")

    train_panel: list[dict[str, str]] = []
    train_indices = []
    rng = np.random.default_rng(SPLIT_SEED)
    for domain in DOMAINS:
        for weather in WEATHER:
            regime = next(key for key, name in corpus.regime_names.items() if name == f"{domain}·{weather}")
            candidates = torch.where(corpus.regimes == regime)[0].numpy()
            if len(candidates) < count:
                raise ValueError(f"Only {len(candidates)} {domain}/{weather} training examples are available.")
            indices = sorted(rng.choice(candidates, count, replace=False).tolist())
            train_indices.extend(indices)
            train_panel.extend(
                {"image": corpus.names[i], "timeofday": domain, "weather": weather, "partition": "train"}
                for i in indices
            )

    test_panel = [row for row in saved["panel"] if row["partition"] == "test"]
    if not test_panel or len({row["image"] for row in train_panel + test_panel}) != len(train_panel) + len(test_panel):
        raise ValueError("Expected distinct train and fixed test images.")
    img_size = int(original["config"]["img_size"])
    transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    test_images = torch.empty((len(test_panel), 3, img_size, img_size))
    for i, row in enumerate(test_panel):
        with Image.open(root / "images/100k/val" / row["image"]) as image:
            test_images[i] = transform(image.convert("RGB"))
    images = torch.cat((corpus.images[train_indices], test_images))
    ids = [rows[f"images/100k/train/{row['image']}"] for row in train_panel] + [
        rows[f"images/100k/val/{row['image']}"] for row in test_panel
    ]
    probabilities = np.load(probability_path, allow_pickle=False, mmap_mode="r")
    return train_panel + test_panel, images, torch.from_numpy(probabilities[ids].copy())


def _split(domains: np.ndarray, labels: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    strata = np.asarray([f"{domain}/{label}" for domain, label in zip(domains, labels, strict=True)])
    pool, test = train_test_split(np.arange(len(labels)), test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=strata)
    cells = {
        (domain, label): pool[(domains[pool] == domain) & (labels[pool] == label)]
        for domain in DOMAINS
        for label in range(len(WEATHER))
    }
    count = min(map(len, cells.values()))
    if count == 0:
        raise ValueError("Every weather class must occur in both training domains.")
    rng = np.random.default_rng(SPLIT_SEED)
    train = {
        domain: np.sort(np.concatenate([rng.choice(cells[domain, label], count, replace=False) for label in range(2)]))
        for domain in DOMAINS
    }
    return train, {domain: np.sort(test[domains[test] == domain]) for domain in DOMAINS}


@torch.no_grad()  # type: ignore[misc]
def _encode(
    cfg: DictConfig, run: dict[str, Any], checkpoint: dict[str, Any], images: torch.Tensor, probabilities: torch.Tensor
) -> np.ndarray:
    arm = "gmm_capped" if run["arm"] == ALL_LAYERS_ARM else run["arm"]
    method = build_arm(cfg, run["seed"], arm)
    method.load_state_dict(checkpoint["model"])
    method.eval()
    features = []
    batch_size = int(cfg.batch_size)
    for batch, posterior in zip(images.split(batch_size), probabilities.split(batch_size), strict=True):
        with _conditioning(method, posterior):
            features.append(method.encode(batch).cpu())
    return torch.cat(features).numpy()


def main() -> None:
    """Evaluate every final checkpoint from the original study and the all-layer ablation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("outputs/m6_3_0"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gmm-dir", type=Path)
    parser.add_argument("--bdd-root", type=Path)
    parser.add_argument(
        "--train-per-class", type=int, help="Fit on this many M6.3 train-cache images per domain/class."
    )
    parser.add_argument("--fixed-test-protocol", type=Path, default=Path("outputs/m6_4_0/protocol.json"))
    args = parser.parse_args()
    if args.train_per_class is not None and args.train_per_class < 1:
        parser.error("--train-per-class must be positive")
    output = args.output or Path("outputs/m6_4_0")
    if args.train_per_class is not None:
        output = args.output or output / f"train_{args.train_per_class}"
        if (output / "protocol.json").resolve() == args.fixed_test_protocol.resolve():
            parser.error("Use a separate output directory for the fixed-test run.")
    reference = args.reference
    original = json.loads((reference / "protocol.json").read_text())
    all_layers = json.loads((reference / ALL_LAYERS_DIR / "protocol.json").read_text())
    protocols = dict.fromkeys(ARMS, original) | {ALL_LAYERS_ARM: all_layers}
    comparison = read_comparison(reference)
    seeds = original["config"]["seeds"]
    runs = sorted(comparison["runs"], key=lambda row: (list(protocols).index(row["arm"]), row["seed"]))
    if [(run["arm"], run["seed"]) for run in runs] != [(arm, seed) for arm in protocols for seed in sorted(seeds)]:
        raise ValueError("Expected the same complete set of seeds for every M6.3 variant.")

    provenance = original["conditioning"]
    gmm_dir = args.gmm_dir or Path(provenance["gmm_directory"])
    torch.set_num_threads(int(original["config"]["threads"]))
    split_metadata: dict[str, Any]
    if args.train_per_class is None:
        names, domains, labels, images, probabilities = _load_panel(
            gmm_dir, args.bdd_root, int(original["config"]["img_size"]), provenance
        )
        train, test = _split(domains, labels)
        panel = [
            {"image": names[i], "timeofday": domains[i], "weather": WEATHER[labels[i]], "partition": partition}
            for partition, split in (("train", train), ("test", test))
            for indices in split.values()
            for i in indices
        ]
        selected = np.sort(np.concatenate([*train.values(), *test.values()]))
        images, probabilities = images[selected], probabilities[selected]
        labels = labels[selected]
        train = {domain: np.searchsorted(selected, ids) for domain, ids in train.items()}
        test = {domain: np.searchsorted(selected, ids) for domain, ids in test.items()}
        split_metadata = {"test_fraction": TEST_SIZE}
    else:
        panel, images, probabilities = _load_train_probe_panel(
            original, gmm_dir, args.bdd_root, args.fixed_test_protocol, args.train_per_class
        )
        split_metadata = {
            "train_per_class_per_domain": args.train_per_class,
            "train_cache": original["data"],
            "fixed_test_protocol_sha256": digest_file(args.fixed_test_protocol),
        }
        train, test = (
            {
                domain: np.asarray(
                    [i for i, row in enumerate(panel) if row["partition"] == partition and row["timeofday"] == domain]
                )
                for domain in DOMAINS
            }
            for partition in ("train", "test")
        )
        labels = np.asarray([WEATHER.index(row["weather"]) for row in panel])
    if len({row["image"] for row in panel}) != len(panel):
        raise ValueError("Probe train and test sets overlap.")
    logger.info("Selected {} train/test images", len(panel))

    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "protocol.json",
        {
            "experiment": "M6.4.0",
            "split_seed": SPLIT_SEED,
            **split_metadata,
            "reference_protocols": {arm: protocol["protocol_hash"] for arm, protocol in protocols.items()},
            "gmm_manifest_sha256": provenance["gmm_manifest_sha256"],
            "gmm_probability_sha256": provenance["gmm_probability_sha256"],
            "versions": {"torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__},
            "panel": panel,
        },
    )
    results = []
    for run in runs:
        arm, seed = run["arm"], run["seed"]
        expected = protocols[arm]
        path = run_directory(reference, run) / "latest.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["protocol_hash"] != expected["protocol_hash"] or checkpoint["step"] != run["steps"]:
            raise ValueError(f"{path} is not the final checkpoint for its protocol.")
        features = _encode(OmegaConf.create(expected["config"]), run, checkpoint, images, probabilities)
        scores = score_transfer(features, labels, train, test)
        gaps = {metric: transfer_gap(scores, metric) for metric in TRANSFER_METRICS}
        results.append(
            {
                "arm": arm,
                "seed": seed,
                "step": checkpoint["step"],
                "sha256": digest_file(path),
                "scores": scores,
                "transfer_gap": gaps,
            }
        )
        logger.info("{} seed {}: rainy F1 transfer gap {:+.3f}", arm, seed, gaps["rainy_f1"])

    summary = {}
    for arm in protocols:
        rows = [row for row in results if row["arm"] == arm]
        summary[arm] = summarize_transfer(rows)
    write_json(output / "results.json", {"runs": results, "summary": summary})


if __name__ == "__main__":
    main()
