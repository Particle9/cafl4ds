"""Compare a pretrained TinyViT encoder with its matched random-init baseline on STL-10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cafl4ds.data.sources import STL10Source
from cafl4ds.data.streams import EraStream
from cafl4ds.measurements import knn_probe, linear_probe
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.ssl.base import apply_encoder_init


def _encoder(seed: int, checkpoint: Path | None) -> TinyViTEncoder:
    torch.manual_seed(seed)
    encoder = TinyViTEncoder(img_size=64, patch_size=8, embed_dim=96, depth=4, num_heads=3, mlp_ratio=2.0)
    if checkpoint is not None:
        apply_encoder_init(encoder, "pretrained", checkpoint)
    return encoder


def main() -> None:
    """Run the two preregistered support/query competence reads and print JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source = STL10Source(str(args.data_root), split="train", img_size=64)
    result: dict[str, dict[str, float]] = {}
    for support, query in ((20, 20), (300, 100)):
        stream = EraStream(
            source,
            batch_size=32,
            order="iid",
            support_per_class=support,
            query_per_class=query,
            era_eval_per_class=1,
            max_train_per_class=1,
            seed=args.seed,
        )
        eval_sets = stream.eval_sets
        key = f"support_{support}_query_{query}"
        result[key] = {}
        for name, checkpoint in (("random", None), ("pretrained", args.checkpoint)):
            encoder = _encoder(args.seed, checkpoint)
            train = (eval_sets.probe_support.images, eval_sets.probe_support.labels)
            test = (eval_sets.probe_query.images, eval_sets.probe_query.labels)
            result[key][f"{name}_knn"] = knn_probe(encoder.embed, train, test, k=20)
            result[key][f"{name}_linear"] = linear_probe(encoder.embed, train, test)
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
