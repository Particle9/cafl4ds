"""Study-only selectors with traced decisions and observational model scoring."""

from __future__ import annotations

import random
import time
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch

from cafl4ds.data.streams import StreamBatch
from cafl4ds.filters.base import Filter, FilterContext


@contextmanager
def observational_score(ctx: FilterContext, seed: int) -> Iterator[None]:
    """Isolate CPU selection randomness and keep BatchNorm buffers unchanged."""
    method = ctx.method
    if next(method.parameters()).device.type != "cpu":
        raise ValueError("P1.4.0 observational scoring is certified for local CPU only")
    modes = [(module, module.training) for module in method.modules()]
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % (2**32))
            method.eval()
            try:
                with torch.no_grad():
                    yield
            finally:
                for module, was_training in modes:
                    module.training = was_training
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


class StudyAcceptAll(Filter):
    """Accept every incoming example and expose a uniform trace interface."""

    def __init__(self) -> None:
        """Initialize the selection trace."""
        self.last_trace: dict[str, object] = {}

    def select(self, batch: StreamBatch, ctx: FilterContext) -> torch.Tensor:
        """Pass through every current row and trace the full budget."""
        del ctx
        self.last_trace = {
            "current_rows": list(range(batch.images.shape[0])),
            "replay_events": [],
            "trained": int(batch.images.shape[0]),
            "selection_seconds": 0.0,
        }
        return batch.images


class StudyLossHalf(Filter):
    """Keep highest-loss examples while leaving training RNG and BN unchanged."""

    def __init__(self, count: int = 64, seed: int = 0) -> None:
        """Set the quota and independent scoring seed."""
        if count < 2:
            raise ValueError("count must be at least two for SimSiam BatchNorm")
        self.count = count
        self.seed = seed
        self.last_trace: dict[str, object] = {}

    def select(self, batch: StreamBatch, ctx: FilterContext) -> torch.Tensor:
        """Score in eval mode under isolated RNG and select the highest-loss rows."""
        if batch.images.shape[0] < self.count:
            raise ValueError("incoming batch is smaller than the registered loss quota")
        start = time.perf_counter()
        with observational_score(ctx, self.seed + ctx.step):
            losses = ctx.method.per_sample_loss(batch.images)
        chosen = torch.topk(losses, self.count).indices.sort().values
        self.last_trace = {
            "current_rows": chosen.tolist(),
            "replay_events": [],
            "trained": self.count,
            "selection_seconds": time.perf_counter() - start,
            "scores": losses.tolist(),
        }
        return batch.images[chosen.to(batch.images.device)]
