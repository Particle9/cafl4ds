"""Regressions for private decoder mixing and restart at the day/night boundary."""

import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from cafl4ds.cluster_routing_experiment import load_data
from cafl4ds.day_night_analysis import analyze
from cafl4ds.day_night_drift import _run as measure_drift
from cafl4ds.day_night_experiment import (
    ARMS,
    build_arm,
    optimizer_groups,
    phase_orders,
    prepare_panel,
    run_arm,
    run_experiment,
)
from cafl4ds.expert_learning_rates import apply_expert_learning_rates
from cafl4ds.ssl.trunk_mixture import TrunkMixtureMAE


def _config() -> DictConfig:
    with initialize_config_dir(config_dir=str(Path("cafl4ds/configs").resolve()), version_base=None):
        return compose(
            config_name="day_night_experiment",
            overrides=[
                "routing_source@source=synthetic",
                "img_size=16",
                "encoder.embed_dim=16",
                "encoder.num_heads=2",
                "ssl.decoder_dim=16",
                "ssl.decoder_heads=2",
                "ssl.decoder_depth=1",
                "support_per_class=4",
                "query_per_class=4",
                "natural_query_size=6",
                "mask_draws=1",
                "batch_size=4",
                "eval_every=2",
                "pilot_images_per_phase=5",
            ],
        )


def test_private_decoders_mix_predictions_and_own_their_optimizer_parameters() -> None:
    """Use the same restore indices for independent decoders, with each decoder at its expert rate."""
    torch.set_num_threads(1)
    cfg = _config()
    method = build_arm(cfg, 0, "gmm_full_private_decoder")
    assert isinstance(method, TrunkMixtureMAE)
    views, noise = torch.rand(2, 3, 16, 16), torch.rand(2, 4)
    probabilities = torch.tensor([[0.75, 0.2, 0.05, 0.0], [0.85, 0.1, 0.05, 0.0]])
    with method.image_routing(probabilities):
        prediction, mask = method.reconstruct(views, noise)
    expected = torch.zeros_like(prediction)
    for index, (encoder, decoder) in enumerate(method.expert_modules()):
        encoded = encoder.forward_with_aux(views, method.mask_ratio, mask_noise=noise)
        torch.testing.assert_close(encoded.mask, mask)
        expected += probabilities[:, index, None, None] * decoder(encoded.tokens, encoded.ids_restore)
    torch.testing.assert_close(prediction, expected)
    assert method.probabilities is None
    opt = torch.optim.AdamW(optimizer_groups(method, "gmm_full_private_decoder", 0.001), lr=0.001)
    rates = apply_expert_learning_rates(opt, {"l0_mean_probabilities": probabilities.mean(0).tolist()})
    torch.testing.assert_close(torch.tensor(rates["l0_expert_lr"]), torch.tensor([0.001, 0.0006, 0.0002, 0.0]))
    all_parameters = [id(p) for group in opt.param_groups for p in group["params"]]
    assert len(set(all_parameters)) == len(all_parameters)
    assert set(all_parameters) == {id(p) for p in method.parameters()}
    for group, (_encoder, decoder) in zip(opt.param_groups, method.expert_modules(), strict=True):
        assert {id(p) for p in decoder.parameters()} <= {id(p) for p in group["params"]}


def test_resume_private_decoder_arm_at_phase_boundary(tmp_path: Path) -> None:
    """Restore optimizer and logs at the exact switch, including the short day batch."""
    torch.set_num_threads(1)
    cfg = _config()
    data = load_data(cfg)
    panel = prepare_panel(cfg, data)
    day, night = phase_orders(data, cfg, 0)
    assert len(day) == len(night) == 5 and set(day.tolist()).isdisjoint(night.tolist())
    arm = "gmm_full_private_decoder"
    reference, resumed = tmp_path / "reference", tmp_path / "resumed"
    complete = run_arm(cfg, data, panel, 0, arm, reference, "test")
    partial = run_arm(cfg, data, panel, 0, arm, resumed, "test", stop_after=2)
    assert partial["step"] == complete["boundary_step"] == 2
    assert (resumed / "pre_night.pt").is_file()
    for name in ("training.jsonl", "evaluation.jsonl"):
        with (resumed / name).open("a") as stream:
            stream.write("uncommitted tail\n")
    result = run_arm(cfg, data, panel, 0, arm, resumed, "test")
    assert result["forgetting_mse"] == complete["forgetting_mse"]
    for name in ("training.jsonl", "evaluation.jsonl"):
        assert (resumed / name).read_bytes() == (reference / name).read_bytes()


def test_paired_experiment_audits_training_and_replays_saved_checkpoints(tmp_path: Path) -> None:
    """Run the configured six arms and verify the audit against the saved inputs and checkpoints."""
    cfg = _config()
    cfg.seeds = [0]
    cfg.threads = 1
    cfg.workers = 1
    result = run_experiment(cfg, tmp_path)
    assert [run["arm"] for run in result["runs"]] == list(ARMS)
    assert result["audit"]["gradient_updates"] == 4 * len(ARMS)
    assert result["audit"]["gmm_rate_vectors"] == 4 * 5
    assert (tmp_path / "reconstruction.png").is_file()

    data = load_data(cfg)
    arm = "gmm_capped"
    folder = tmp_path / "seed_0" / arm
    measure_drift(tmp_path, 0, arm, True, data=data)
    measure_drift(tmp_path, 0, arm, False, data=data)
    checkpoints = json.loads((folder / "drift_checkpoints.json").read_text())["rows"]
    replay = json.loads((folder / "drift.json").read_text())["rows"]
    assert [row["step"] for row in replay] == [0, 2, 3, 4]
    assert [row["step"] for row in checkpoints] == [0, 2, 4]
    assert checkpoints == [row for row in replay if row["step"] in {0, 2, 4}]

    path = folder / "training.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["sample_ids"].reverse()
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="trace differs from the frozen"):
        analyze(tmp_path, data.probabilities)
