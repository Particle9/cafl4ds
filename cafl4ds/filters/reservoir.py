"""Reservoir sampling + experience replay — the replay knob (Vitter 1985; baseline B1).

A fixed-capacity buffer maintained by **Vitter's Algorithm R**: after ``N`` frames have passed,
each is present with equal probability ``capacity / N``, so the buffer is a uniform random
sample of the whole (single-pass, unbounded) stream — no matter how correlated the arrival
order. Each step the model trains on the incoming frames **plus** a random replay draw from the
buffer, mixing past eras back into a correlated stream. That interleaving is the "dumb"
protective bar (B1): replay alone, no informativeness, partly counters forgetting.

The buffer is the **replay** role of the ``A`` factor (see :mod:`cafl4ds.filters.base`): it
runs after any admission filters, so what it stores and replays is the *admitted* stream —
upstream :class:`~cafl4ds.filters.dedup.SemDeDup` gives reservoir + dedup (baseline B1.5).
Sampling uses a private, seeded generator so a run is reproducible independently of whatever
else consumes the global RNG.
"""

from __future__ import annotations

import torch

from cafl4ds.data.streams import StreamBatch
from cafl4ds.filters.base import Filter, FilterContext, ReplayBuffer


class ReservoirReplay(ReplayBuffer):
    """Uniform reservoir buffer (Algorithm R) with per-step experience replay."""

    def __init__(self, capacity: int = 256, replay_batch: int = 32, seed: int = 0) -> None:
        """Configure the reservoir.

        Args:
            capacity: Maximum number of frames held in the buffer.
            replay_batch: Number of past frames replayed alongside each incoming batch (a
                uniform draw, without replacement, from the current buffer; clamped to the
                buffer's size, so early steps replay fewer).
            seed: Seed for the buffer's private RNG (reservoir replacement + replay sampling).

        Raises:
            ValueError: If ``capacity`` or ``replay_batch`` is not positive.
        """
        if capacity < 1 or replay_batch < 1:
            raise ValueError(f"capacity and replay_batch must be >= 1; got {capacity}, {replay_batch}.")
        self.capacity = capacity
        self.replay_batch = replay_batch
        self._buffer: list[torch.Tensor] = []
        self._seen = 0  # total frames observed so far (Algorithm R's running count N)
        self._gen = torch.Generator().manual_seed(seed)  # CPU generator; indices only

    def observe(self, images: torch.Tensor, ctx: FilterContext) -> torch.Tensor:
        """Replay from the buffer, then ingest the incoming frames (Algorithm R).

        Replay is sampled from the buffer *before* the incoming frames are ingested, so it is
        drawn from genuine history (never the just-arrived frames), then each incoming frame
        updates the reservoir.

        Args:
            images: The admitted images ``[K, C, H, W]`` for this step.
            ctx: Unused (reservoir replay is model-agnostic and random).

        Returns:
            The training batch ``[K + R, C, H, W]``: the incoming frames followed by ``R``
            replayed frames (``R = min(replay_batch, buffer size before this step)``).
        """
        del ctx  # reservoir selection is model-agnostic
        replay = self._sample_replay()
        for i in range(images.shape[0]):
            self._ingest(images[i])
        return images if replay is None else torch.cat([images, replay], dim=0)

    def _sample_replay(self) -> torch.Tensor | None:
        """Draw a uniform replay batch from the current buffer (without replacement).

        Returns:
            A ``[R, C, H, W]`` stack of replayed frames, or ``None`` if the buffer is empty.
        """
        if not self._buffer:
            return None
        r = min(self.replay_batch, len(self._buffer))
        idx = torch.randperm(len(self._buffer), generator=self._gen)[:r].tolist()
        return torch.stack([self._buffer[i] for i in idx], dim=0)

    def _ingest(self, image: torch.Tensor) -> None:
        """Update the reservoir with one frame via Vitter's Algorithm R.

        While the buffer is not full every frame is stored; once full, the ``N``-th frame
        (0-based index ``N``) replaces a uniformly-random slot with probability
        ``capacity / (N + 1)`` — the invariant that keeps the buffer a uniform sample.

        Args:
            image: A single frame ``[C, H, W]`` to (maybe) store; a detached clone is kept.
        """
        item = image.detach().clone()
        if len(self._buffer) < self.capacity:
            self._buffer.append(item)
        else:
            j = int(torch.randint(0, self._seen + 1, (1,), generator=self._gen).item())
            if j < self.capacity:
                self._buffer[j] = item
        self._seen += 1


class FixedBudgetReservoir(Filter):
    """Replace some current examples with past arrivals at fixed update size."""

    def __init__(
        self,
        *,
        incoming_count: int = 128,
        replay_count: int = 64,
        capacity: int = 256,
        seed: int = 0,
        arrival_batches: tuple[tuple[int, ...], ...],
    ) -> None:
        """Configure a uniform arrival-event reservoir with a fixed training budget."""
        if incoming_count < 2 or replay_count < 1 or replay_count >= incoming_count or capacity < replay_count:
            raise ValueError("invalid fixed-budget reservoir dimensions")
        self.incoming_count = incoming_count
        self.replay_count = replay_count
        self.capacity = capacity
        self.arrival_batches = arrival_batches
        self._buffer: list[tuple[torch.Tensor, int, int]] = []
        self._seen = 0
        self._gen = torch.Generator().manual_seed(seed)
        self.last_trace: dict[str, object] = {}

    def select(self, batch: StreamBatch, ctx: FilterContext) -> torch.Tensor:
        """Sample only past events, then ingest every current arrival."""
        import time  # noqa: PLC0415 - selection timing kept local to this adapter

        if batch.images.shape[0] != self.incoming_count:
            raise ValueError("fixed-budget replay requires the registered incoming batch size")
        start = time.perf_counter()
        ids = self.arrival_batches[ctx.step % len(self.arrival_batches)]
        if len(ids) != self.incoming_count:
            raise ValueError("arrival ID manifest disagrees with incoming batch")
        r = min(self.replay_count, len(self._buffer))
        replay_rows = torch.randperm(len(self._buffer), generator=self._gen)[:r].tolist()
        current_rows = torch.randperm(self.incoming_count, generator=self._gen)[: self.incoming_count - r]
        current_rows = current_rows.sort().values
        replay = [self._buffer[i] for i in replay_rows]
        selected = batch.images[current_rows.to(batch.images.device)]
        if replay:
            selected = torch.cat((selected, torch.stack([entry[0] for entry in replay]).to(batch.images.device)))
        # Draw from previous history first; every raw arrival is then eligible for storage.
        for row, source_id in enumerate(ids):
            item = (batch.images[row].detach().cpu().clone(), ctx.step, source_id)
            if len(self._buffer) < self.capacity:
                self._buffer.append(item)
            else:
                replacement = int(torch.randint(0, self._seen + 1, (1,), generator=self._gen))
                if replacement < self.capacity:
                    self._buffer[replacement] = item
            self._seen += 1
        self.last_trace = {
            "current_rows": current_rows.tolist(),
            "replay_events": [[step, source_id] for _, step, source_id in replay],
            "trained": self.incoming_count,
            "selection_seconds": time.perf_counter() - start,
        }
        return selected
