"""Fixed-count, reproducible admission for matched selection experiments."""

from __future__ import annotations

import time

import torch

from cafl4ds.data.streams import StreamBatch
from cafl4ds.filters.base import Filter, FilterContext


class RandomCount(Filter):
    """Keep exactly ``count`` uniformly drawn current examples per update."""

    def __init__(self, count: int = 64, seed: int = 0) -> None:
        """Set the fixed training quota and independent sampling generator."""
        if count < 2:
            raise ValueError("count must be at least two for SimSiam BatchNorm")
        self.count = count
        self._gen = torch.Generator().manual_seed(seed)
        self.last_trace: dict[str, object] = {}

    def select(self, batch: StreamBatch, ctx: FilterContext) -> torch.Tensor:
        """Sample the registered number of distinct current rows and trace them."""
        del ctx
        if batch.images.shape[0] < self.count:
            raise ValueError("incoming batch is smaller than the registered random quota")
        start = time.perf_counter()
        chosen = torch.randperm(batch.images.shape[0], generator=self._gen)[: self.count].sort().values
        self.last_trace = {
            "current_rows": chosen.tolist(),
            "replay_events": [],
            "trained": self.count,
            "selection_seconds": time.perf_counter() - start,
        }
        return batch.images[chosen.to(batch.images.device)]
