"""Shared scoring and matched initialization for the frozen weather-probe studies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cafl4ds.cluster_routing_analysis import _interval
from cafl4ds.day_night_experiment import _dense_encoder
from cafl4ds.ssl.mae import MAE

DOMAINS = ("daytime", "night")
WEATHER = ("clear", "rainy")
TRANSFER_METRICS = ("accuracy", "balanced_accuracy", "rainy_f1")
DENSE_ARM = "dense"
DENSE_REFERENCE_ARM = "gmm_capped"


def score_transfer(
    features: np.ndarray, labels: np.ndarray, train: dict[str, np.ndarray], test: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Fit one linear weather probe per source domain and score every target domain."""
    scores = {}
    for source, support in train.items():
        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        classifier.fit(features[support], labels[support])
        for target, query in test.items():
            expected = labels[query]
            predicted = classifier.predict(features[query])
            scores[f"{source}_to_{target}"] = {
                "accuracy": accuracy_score(expected, predicted),
                "balanced_accuracy": balanced_accuracy_score(expected, predicted),
                "rainy_f1": f1_score(expected, predicted, pos_label=1, zero_division=0),
                "confusion_matrix": confusion_matrix(expected, predicted, labels=[0, 1]).tolist(),
            }
    return scores


def transfer_gap(scores: dict[str, Any], metric: str) -> float:
    """Return mean within-domain performance minus mean cross-domain performance."""
    day, night = DOMAINS
    return float(
        (
            scores[f"{day}_to_{day}"][metric]
            + scores[f"{night}_to_{night}"][metric]
            - scores[f"{day}_to_{night}"][metric]
            - scores[f"{night}_to_{day}"][metric]
        )
        / 2
    )


def summarize_transfer(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate directional scores and transfer gaps across paired model seeds."""
    return {
        "scores": {
            direction: {
                metric: _interval([run["scores"][direction][metric] for run in runs]) for metric in TRANSFER_METRICS
            }
            for direction in runs[0]["scores"]
        },
        "transfer_gap": {
            metric: _interval([run["transfer_gap"][metric] for run in runs]) for metric in TRANSFER_METRICS
        },
    }


def build_dense_control(cfg: DictConfig, seed: int, arm: str, *, initial: Path) -> MAE:
    """Initialize a dense MAE from the matching shared weights and first expert."""
    if arm != DENSE_ARM:
        raise ValueError(f"Unsupported dense-control arm {arm} for seed {seed}.")
    encoder = _dense_encoder(cfg)
    method: MAE = instantiate(cfg.ssl, encoder=encoder)
    reference = torch.load(initial, map_location="cpu", weights_only=True)["model"]
    # Copy shared tensors and the first expert in each routed MLP to the dense equivalent.
    state = {
        key: reference[key] if key in reference else reference[key.replace(".mlp.", ".mlp.experts.0.")]
        for key in method.state_dict()
    }
    method.load_state_dict(state)
    method.to(cfg.device)
    return method
