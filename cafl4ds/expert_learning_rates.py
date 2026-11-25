"""Posterior-dependent learning rates outside AdamW's moment normalization."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from cafl4ds.models.fixed_mixture import FixedImageMixture
from cafl4ds.models.vit import MoEMlp
from cafl4ds.ssl.mae import MAE


def expert_parameter_groups(method: MAE, base_lr: float) -> list[dict[str, Any]]:
    """Partition parameters once into shared/router parameters and each layer's experts."""
    if not math.isfinite(base_lr) or base_lr <= 0:
        raise ValueError("The expert base learning rate must be finite and positive.")
    groups: list[dict[str, Any]] = []
    expert_ids: set[int] = set()
    for layer, block in enumerate(method.encoder.blocks):
        if not isinstance(block.mlp, MoEMlp | FixedImageMixture):
            continue
        for expert_id, expert in enumerate(block.mlp.experts):
            parameters = list(expert.parameters())
            ids = {id(parameter) for parameter in parameters}
            if expert_ids & ids:
                raise ValueError("An expert parameter is shared by multiple optimizer groups.")
            expert_ids.update(ids)
            groups.append(
                {
                    "params": parameters,
                    "lr": base_lr,
                    "base_lr": base_lr,
                    "expert_layer": layer,
                    "expert_index": expert_id,
                    "num_experts": len(block.mlp.experts),
                    "name": f"l{layer}_expert_{expert_id}",
                }
            )
    if not groups:
        raise ValueError("Expert-specific learning rates require MoE layers.")
    shared = [parameter for parameter in method.parameters() if id(parameter) not in expert_ids]
    return [{"params": shared, "lr": base_lr, "base_lr": base_lr, "name": "shared_and_routers"}, *groups]


def apply_expert_learning_rates(
    optimizer: torch.optim.Optimizer, readout: Mapping[str, Any], cap: float | None = 1.0
) -> dict[str, Any]:
    """Scale the final AdamW step by E*q, with an optional cap and unchanged moments."""
    if cap is not None and (not math.isfinite(cap) or cap <= 0):
        raise ValueError("An expert learning-rate cap must be finite and positive, or None.")
    rows: dict[str, Any] = {}
    for group in optimizer.param_groups:
        base_lr = float(group["base_lr"])
        if "expert_layer" not in group:
            group["lr"] = base_lr
            rows["shared_and_router_lr"] = base_lr
            continue
        layer, expert, count = int(group["expert_layer"]), int(group["expert_index"]), int(group["num_experts"])
        probabilities = readout[f"l{layer}_mean_probabilities"]
        if len(probabilities) != count:
            raise ValueError("Training readout does not match the expert optimizer groups.")
        probability = float(probabilities[expert])
        if not math.isfinite(probability) or probability < 0:
            raise ValueError("Expert probability mass must be finite and nonnegative.")
        scale = count * probability if cap is None else min(cap, count * probability)
        group["lr"] = base_lr * scale
        rows.setdefault(f"l{layer}_expert_lr_scale", [0.0] * count)[expert] = scale
        rows.setdefault(f"l{layer}_expert_lr", [0.0] * count)[expert] = group["lr"]
    return rows


def snapshot_experts(optimizer: torch.optim.Optimizer) -> dict[str, list[torch.Tensor]]:
    """Copy expert parameters immediately before the optimizer step."""
    return {
        group["name"]: [parameter.detach().clone() for parameter in group["params"]]
        for group in optimizer.param_groups
        if "expert_layer" in group
    }


def expert_update_norms(
    optimizer: torch.optim.Optimizer, before: Mapping[str, list[torch.Tensor]]
) -> dict[str, list[float]]:
    """Measure the actual parameter-step L2 norm, including learning-rate-scaled decay."""
    rows: dict[str, list[float]] = {}
    for group in optimizer.param_groups:
        if "expert_layer" not in group:
            continue
        layer, expert, count = int(group["expert_layer"]), int(group["expert_index"]), int(group["num_experts"])
        squares = [
            (parameter.detach() - old).square().sum()
            for parameter, old in zip(group["params"], before[group["name"]], strict=True)
        ]
        rows.setdefault(f"l{layer}_expert_update_norm", [0.0] * count)[expert] = float(
            torch.stack(squares).sum().sqrt()
        )
    return rows
