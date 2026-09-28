"""P1.4.0 invariants that make the diet comparisons interpretable."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from cafl4ds.collapse_diet import (
    TensorSource,
    _splits,
    analyze_confirmation_diet,
    bootstrap_interval,
    make_stream,
    run_stage,
    seed_components,
    sign_flip_pvalue,
)
from cafl4ds.data.streams import StreamBatch
from cafl4ds.filters.base import FilterContext
from cafl4ds.filters.random import RandomCount
from cafl4ds.filters.reservoir import FixedBudgetReservoir
from cafl4ds.filters.study import StudyLossHalf


def test_fixed_splits_and_equal_batches_across_orderings() -> None:
    """The dose sweep changes order, while source IDs and update counts stay fixed."""
    labels = torch.arange(10).repeat_interleave(484)
    source = TensorSource(torch.zeros(4840, 1, 4, 4), labels)
    values = {
        "batch_size": 128,
        "train_per_class": 384,
        "support_per_class": 40,
        "query_per_class": 50,
        "era_eval_per_class": 10,
    }
    seeds = seed_components(3)
    splits = _splits(source, values, seeds)
    streams = {name: make_stream(source, values, seeds, splits, name) for name in ("iid", "b128", "b256", "full")}
    expected = set(streams["iid"].ordered_ids)
    for stream in streams.values():
        assert len(stream) == 30
        assert all(len(batch) == 128 for batch in stream.batch_ids)
        assert set(stream.ordered_ids) == expected
        assert len(stream.ordered_ids) == len(expected) == 3840
    for cls in range(10):
        groups = (splits.support[cls], splits.query[cls], splits.era_eval[cls], splits.train[cls])
        assert len(set().union(*(set(group) for group in groups))) == 484
    assert streams["b128"].class_order == streams["b256"].class_order == streams["full"].class_order


class _ScoringMethod(torch.nn.Module):
    """A tiny stochastic model that would mutate BN if scored in training mode."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(3)

    def per_sample_loss(self, images: torch.Tensor) -> torch.Tensor:
        return self.bn(images.flatten(1)).mean(dim=1) + torch.rand(images.shape[0])


def test_loss_scoring_is_observational() -> None:
    """Loss scoring selects a quota without changing state, mode, or training RNG."""
    method = _ScoringMethod()
    method.train()
    images = torch.rand(8, 3, 1, 1)
    batch = StreamBatch(images=images, era=0, step=0)
    state = copy.deepcopy(method.state_dict())
    mode = [module.training for module in method.modules()]
    rng = torch.random.get_rng_state().clone()
    selector = StudyLossHalf(count=4, seed=12)
    selected = selector.select(batch, FilterContext(method=method, step=0))  # type: ignore[arg-type]
    assert selected.shape[0] == 4
    assert len(set(selector.last_trace["current_rows"])) == 4  # type: ignore[arg-type]
    assert torch.equal(rng, torch.random.get_rng_state())
    assert mode == [module.training for module in method.modules()]
    assert all(torch.equal(before, method.state_dict()[key]) for key, before in state.items())
    assert torch.equal(selected, images[selector.last_trace["current_rows"]])


def test_fixed_budget_replay_uses_only_prior_arrivals() -> None:
    """Every replay event predates its update and startup still trains eight examples."""
    batches = ((0, 1, 2, 3, 4, 5, 6, 7), (8, 9, 10, 11, 12, 13, 14, 15))
    selector = FixedBudgetReservoir(incoming_count=8, replay_count=4, capacity=8, arrival_batches=batches, seed=4)
    for step, ids in enumerate(batches):
        images = torch.tensor(ids, dtype=torch.float32).view(8, 1, 1, 1)
        selected = selector.select(
            StreamBatch(images, era=0, step=step), FilterContext(method=_ScoringMethod(), step=step)
        )  # type: ignore[arg-type]
        trace = selector.last_trace
        assert len(selected) == 8
        assert trace["trained"] == 8
        assert all(event[0] < step for event in trace["replay_events"])  # type: ignore[union-attr]
        if step == 0:
            assert trace["replay_events"] == []
        else:
            assert len(trace["replay_events"]) == 4  # type: ignore[arg-type]
        expected = [ids[i] for i in trace["current_rows"]]  # type: ignore[union-attr]
        expected += [event[1] for event in trace["replay_events"]]  # type: ignore[union-attr]
        assert selected.flatten().tolist() == expected


def test_random_quota_and_paired_statistics() -> None:
    """Quota and seed-level effects have the registered signs and exact test granularity."""
    selector = RandomCount(count=4, seed=8)
    batch = StreamBatch(torch.arange(8, dtype=torch.float32).view(8, 1, 1, 1), era=0, step=0)
    chosen = selector.select(batch, FilterContext(method=_ScoringMethod(), step=0))  # type: ignore[arg-type]
    assert len(chosen) == 4
    assert len(set(selector.last_trace["current_rows"])) == 4  # type: ignore[arg-type]
    assert sign_flip_pvalue([1.0] * 10) == pytest.approx(2 / 1024)
    assert bootstrap_interval([0.2] * 5) == pytest.approx((0.2, 0.2))


def test_confirmation_distinguishes_quality_geometry_and_null() -> None:
    """The scientific label needs both paired quality harm and paired geometry loss."""

    def comparison(accuracy: float, rank: float) -> dict[str, object]:
        return {
            "arms": {
                "live": {
                    "health": [{"epoch": epoch, "linear_acc": accuracy, "rankme_proj": rank} for epoch in range(36, 41)]
                }
            }
        }

    seeds = list(range(100, 110))
    results = {
        (seed, ordering, "accept_all", False): comparison(
            0.8 if ordering == "iid" else 0.6,
            40.0 if ordering == "iid" else 20.0,
        )
        for seed in seeds
        for ordering in ("iid", "full")
    }
    verdict = analyze_confirmation_diet(results, seeds, "full", draws=100, quality_margin=0.03, geometry_ratio=1.25)  # type: ignore[arg-type]
    assert verdict["classification"] == "geometry_associated_degradation"
    for seed in seeds:
        results[(seed, "full", "accept_all", False)] = comparison(0.8, 20.0)
    geometry_only = analyze_confirmation_diet(
        results, seeds, "full", draws=100, quality_margin=0.03, geometry_ratio=1.25
    )  # type: ignore[arg-type]
    assert geometry_only["classification"] == "geometry_change_without_confirmed_quality_harm"
    for seed in seeds:
        results[(seed, "full", "accept_all", False)] = comparison(0.8, 40.0)
    null = analyze_confirmation_diet(results, seeds, "full", draws=100, quality_margin=0.03, geometry_ratio=1.25)  # type: ignore[arg-type]
    assert null["classification"] == "not_confirmed"


def test_cpu_smoke_artifacts_and_clocks(tmp_path: Path) -> None:
    """The complete synthetic stage logs pristine and epoch-end states and selection traces."""
    config_dir = Path(__file__).resolve().parents[2] / "cafl4ds" / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(
            config_name="collapse_diet",
            overrides=[
                "stage=smoke",
                "profile=smoke",
                "data_kind=synthetic",
                "num_threads=1",
                f"output_root={tmp_path}",
            ],
        )
    summary = run_stage(config)
    assert summary["runs"] == 6
    initial_hashes = set()
    for run_dir in tmp_path.glob("smoke_seed0_*"):
        comparison = json.loads((run_dir / "comparison.json").read_text())
        assert comparison["gate"]["passed"]
        initial_hashes.add(comparison["collapse_diet"]["initial_state_hash"])
        role = "pc" if comparison["collapse_diet"]["pc"] else "live"
        health = comparison["arms"][role]["health"]
        assert [row["optimizer_step"] for row in health] == [0, 8, 16]
        assert [row["epoch"] for row in health] == [0, 1, 2]
        trace = json.loads((run_dir / "selection.json").read_text())
        assert len(trace) == 16
        assert trace[0]["raw_seen"] == 8
        assert trace[-1]["raw_seen"] == 128
        assert trace[0]["optimizer_step_before"] == 0
        assert trace[-1]["optimizer_step_before"] == 15
        assert (run_dir / "status.json").exists()
    assert len(initial_hashes) == 1
    assert run_stage(config) == summary  # complete arms are reused only at matching provenance
