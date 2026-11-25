"""Fixed image-level mixtures, using the same experts and dense sum as learned top-E MoE."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

import torch
from torch import nn

from cafl4ds.models.vit import FeedForwardOutput, MoEMlp, mix_all_experts
from cafl4ds.ssl.mae import MAE


class FixedImageMixture(nn.Module):  # type: ignore[misc]
    """A router-free mixture; its temporary conditioning must be supplied for each batch."""

    def __init__(self, experts: nn.ModuleList) -> None:
        """Retain the supplied experts without reinitializing them or constructing a router."""
        super().__init__()
        self.experts = experts
        self.num_experts = len(experts)
        self.probabilities: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, *, deterministic: bool = False) -> FeedForwardOutput:
        """Broadcast the same detached image probabilities over visible patches and CLS."""
        del deterministic
        if self.probabilities is None or self.probabilities.shape != (len(x), self.num_experts):
            raise ValueError("Fixed image routing requires aligned [batch, experts] probabilities.")
        probabilities = self.probabilities[:, None, :].expand(-1, x.shape[1], -1)
        return FeedForwardOutput(mix_all_experts(self.experts, x, probabilities), x.new_zeros(()), probabilities)

    def forward_with_aux(self, x: torch.Tensor, *, deterministic: bool = False) -> FeedForwardOutput:
        """Use the standard module call path so training observers see the actual forward."""
        return cast(FeedForwardOutput, self(x, deterministic=deterministic))


def replace_routers(method: MAE) -> None:
    """Remove learned routers while preserving all encoder, expert and decoder weights."""
    for block in method.encoder.blocks:
        if isinstance(block.mlp, MoEMlp):
            block.mlp = FixedImageMixture(block.mlp.experts)
    method.aux_weight = 0.0


@contextmanager
def image_routing(method: MAE, probabilities: torch.Tensor) -> Iterator[None]:
    """Bind batch conditioning transactionally; never retain the previous batch on failure."""
    layers = [module for module in method.modules() if isinstance(module, FixedImageMixture)]
    if not layers:
        yield
        return
    if probabilities.ndim != 2 or probabilities.shape[1] != layers[0].num_experts:
        raise ValueError("Expected image-by-expert probabilities.")
    if not bool(torch.isfinite(probabilities).all()) or bool((probabilities < 0).any()):
        raise ValueError("Fixed routing probabilities must be finite and nonnegative.")
    sums = probabilities.sum(-1, keepdim=True)
    if bool((sums <= 0).any()):
        raise ValueError("Fixed routing probabilities must have positive mass per image.")
    probabilities = (probabilities.detach() / sums.detach()).to(next(method.parameters()).device)
    previous = [layer.probabilities for layer in layers]
    for layer in layers:
        layer.probabilities = probabilities
    try:
        yield
    finally:
        for layer, value in zip(layers, previous, strict=True):
            layer.probabilities = value
