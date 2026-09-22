"""Measure within-batch redundancy in the exact P1.2 BDD stream before choosing SemDeDup.

The diagnostic uses the init-matched live encoder and reports adjacent-image cosine similarity,
each image's maximum similarity to an earlier image in its batch, and SemDeDup keep rates over a
small threshold grid. It does not train or modify the model.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from cafl4ds.data.attributes import AttributeSource
from cafl4ds.data.regime import RegimeStream
from cafl4ds.filters.dedup import semantic_dedup_keep
from cafl4ds.ssl.base import SSLMethod, apply_encoder_init

ROOT = Path(__file__).resolve().parents[1]


def _quantiles(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values)
    return {
        name: float(torch.quantile(tensor, q))
        for name, q in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99))
    }


def measure(bdd_root: str, seeds: list[int], thresholds: list[float]) -> dict[str, object]:
    """Measure redundancy across seeds using the configured P1.2 stream and initial encoder."""
    config_dir = ROOT / "cafl4ds" / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        base = compose(
            config_name="adaptation_bdd_replay",
            overrides=[f"bdd_root={bdd_root}", f"seeds={seeds}"],
        )
    source: AttributeSource = instantiate(base.data)
    per_seed: list[dict[str, object]] = []
    for seed in seeds:
        config = OmegaConf.create(OmegaConf.to_container(base, resolve=False))
        config.seed = seed
        torch.manual_seed(seed)
        stream: RegimeStream = instantiate(config.stream, source=source)
        method: SSLMethod = instantiate(config.ssl, encoder=instantiate(config.encoder))
        apply_encoder_init(method.encoder, config.init.mode, config.init.checkpoint)
        method.eval()
        adjacent: list[float] = []
        max_prior: list[float] = []
        kept = dict.fromkeys(thresholds, 0)
        total = 0
        with torch.no_grad():
            for batch in stream:
                embeddings = torch.nn.functional.normalize(method.encode(batch.images), dim=1)
                total += embeddings.shape[0]
                if embeddings.shape[0] > 1:
                    adjacent.extend((embeddings[1:] * embeddings[:-1]).sum(dim=1).tolist())
                for i in range(1, embeddings.shape[0]):
                    max_prior.append(float((embeddings[i] @ embeddings[:i].T).max()))
                for threshold in thresholds:
                    kept[threshold] += int(semantic_dedup_keep(embeddings, threshold).numel())
        per_seed.append(
            {
                "seed": seed,
                "images": total,
                "adjacent_cosine": _quantiles(adjacent),
                "max_prior_cosine": _quantiles(max_prior),
                "keep_fraction": {str(t): kept[t] / total for t in thresholds},
            }
        )
    return {
        "schema_version": 1,
        "seeds": per_seed,
        "mean_keep_fraction": {
            str(t): statistics.fmean(float(row["keep_fraction"][str(t)]) for row in per_seed)  # type: ignore[index]
            for t in thresholds
        },
        "interpretation": "Initial-encoder, within-batch diagnostic; no training was performed.",
    }


def main() -> None:
    """Parse CLI arguments, measure the stream, and write the diagnostic JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bdd-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[401, 409, 419])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.9, 0.95, 0.99, 0.995, 0.999])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = measure(args.bdd_root, args.seeds, args.thresholds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
