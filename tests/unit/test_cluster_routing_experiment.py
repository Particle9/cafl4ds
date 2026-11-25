"""Fixed conditioning, paired initialization, real-step routing and resumable MAE comparisons."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from cafl4ds.cluster_routing_analysis import analyze, read_rows
from cafl4ds.cluster_routing_experiment import (
    StepRoutingReadout,
    build_method,
    evaluate,
    forgetting_summary,
    load_data,
    run_arm,
    run_experiment,
)
from cafl4ds.models.fixed_mixture import FixedImageMixture, image_routing


def _config() -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(Path("cafl4ds/configs").resolve())):
        cfg = compose(config_name="cluster_routing_experiment", overrides=["routing_source@source=synthetic"])
    cfg.img_size = 16
    cfg.encoder.embed_dim = 16
    cfg.encoder.depth = 2
    cfg.encoder.num_heads = 2
    cfg.ssl.decoder_dim = 16
    cfg.ssl.decoder_heads = 2
    cfg.ssl.decoder_depth = 1
    cfg.batch_size = 8
    cfg.support_per_class = 4
    cfg.query_per_class = 4
    cfg.natural_query_size = 4
    cfg.mask_draws = 1
    cfg.eval_every = 2
    cfg.seeds = [0, 1]
    cfg.workers = 1
    cfg.threads = 1
    torch.set_num_threads(1)
    return cfg


def test_shared_initial_weights_and_fixed_router_removal() -> None:
    """Only learned routers differ initially; the GMM arm cannot update a hidden router."""
    cfg = _config()
    learned, left_hash = build_method(cfg, 0, "learned")
    fixed, right_hash = build_method(cfg, 0, "gmm")
    assert left_hash == right_hash
    assert any(".router." in name for name, _ in learned.named_parameters())
    assert not any(".router." in name for name, _ in fixed.named_parameters())
    for key, value in fixed.state_dict().items():
        torch.testing.assert_close(value, learned.state_dict()[key], rtol=0, atol=0)
    assert build_method(cfg, 1, "learned")[1] != left_hash


def test_fixed_mixture_is_weighted_sum_and_conditioning_has_no_gradient() -> None:
    """The supplied weights affect expert gradients but receive no gradients themselves."""
    cfg = _config()
    method, _ = build_method(cfg, 0, "gmm")
    layer = next(module for module in method.modules() if isinstance(module, FixedImageMixture))
    x = torch.randn(4, 3, 16, requires_grad=True)
    probabilities = torch.tensor(
        [[0.4, 0.3, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1], [0, 1, 0, 0.0]], requires_grad=True
    )
    observer = StepRoutingReadout(method)
    with image_routing(method, probabilities):
        result = layer(x)
        expected = sum(expert(x) * probabilities[:, i, None, None].detach() for i, expert in enumerate(layer.experts))
        torch.testing.assert_close(result.features, expected)
        torch.testing.assert_close(result.routing_probs, probabilities.detach()[:, None, :].expand(-1, 3, -1))
        result.features.square().sum().backward()
    observer.close()
    assert probabilities.grad is None and result.aux_loss == 0
    assert all(expert.fc1.weight.grad is not None for expert in layer.experts)
    # The minibatch mean favors expert 1, although three of four image argmaxes favor expert 0.
    assert observer.rows["l0_batch_top1_expert"] == 1
    assert observer.rows["l0_image_top1_counts"] == [3, 1, 0, 0]
    assert all(module.probabilities is None for module in method.modules() if isinstance(module, FixedImageMixture))
    with pytest.raises(ValueError, match="aligned"):
        layer(x)
    with pytest.raises(RuntimeError), image_routing(method, probabilities):
        raise RuntimeError("simulated failure")
    assert layer.probabilities is None


def test_evaluation_restores_modes_rng_and_conditioning() -> None:
    """Extra health reads must not alter training, stochastic masks or fixed routing state."""
    cfg = _config()
    data = load_data(cfg)
    method, _ = build_method(cfg, 0, "gmm")
    method.train()
    rng = torch.get_rng_state().clone()
    metrics, features, routes, _ = evaluate(method, data, cfg, 0, None, None, None, True)
    assert method.training and torch.equal(rng, torch.get_rng_state())
    assert set(metrics["regime_mse"]) == {"0", "1"}
    assert all(np.isfinite(metrics[key]) for key in ("uniformity", "alignment", "rankme", "heldout_mse"))
    assert features.shape == (8, 16)
    assert all(module.probabilities is None for module in method.modules() if isinstance(module, FixedImageMixture))
    for values in routes.values():
        torch.testing.assert_close(values, data.probabilities[data.panels["query"]])
    assert not set(data.regime_panel.ids.tolist()) & set(data.panels["train"])


@pytest.mark.parametrize("optimizer_rule", ["constant", "capped", "uncapped"])
@pytest.mark.parametrize("arm", ["learned", "gmm"])
def test_exact_resume_recovers_the_same_training_and_endpoint(tmp_path: Path, arm: str, optimizer_rule: str) -> None:
    """Checkpoint offsets discard uncommitted records and reproduce the uninterrupted result."""
    cfg = _config()
    data = load_data(cfg)
    cfg.expert_lr_scaling = optimizer_rule != "constant"
    OmegaConf.update(cfg, "expert_lr_cap", None if optimizer_rule == "uncapped" else 1.0, force_add=True)
    full = run_arm(cfg, data, 0, arm, tmp_path / "full", "test")
    partial = run_arm(cfg, data, 0, arm, tmp_path / "resume", "test", stop_after=2)
    assert partial["step"] == 2
    with (tmp_path / "resume/training.jsonl").open("a") as stream:
        stream.write('{"uncommitted":true}\n')
    resumed = run_arm(cfg, data, 0, arm, tmp_path / "resume", "test")
    left = torch.load(tmp_path / "full/latest.pt", weights_only=False)
    right = torch.load(tmp_path / "resume/latest.pt", weights_only=False)
    for key, value in left["method"].items():
        torch.testing.assert_close(value, right["method"][key], rtol=0, atol=0)
    assert full["final"] == resumed["final"]
    assert full["forgetting"] == resumed["forgetting"]
    assert (tmp_path / "full/training.jsonl").read_text() == (tmp_path / "resume/training.jsonl").read_text()


def test_complete_paired_ensemble_and_analysis(tmp_path: Path) -> None:
    """Exercise every save/load/evaluation/report path and reject a corrupted paired trace."""
    cfg = _config()
    result = run_experiment(cfg, tmp_path)
    assert len(result["runs"]) == 4
    assert result["replicate_unit"] == "training seed"
    assert all(result["verification"].values())
    for seed in cfg.seeds:
        learned, fixed = (read_rows(tmp_path / f"seed_{seed}" / arm / "training.jsonl") for arm in ("learned", "gmm"))
        assert len(learned) == len(fixed) == 6
        assert all(row["n_images"] == 8 for row in learned)
        assert all(row["auxiliary"] == 0 for row in fixed)
        for row in fixed:
            np.testing.assert_allclose(row["l0_mean_probabilities"], row["l1_mean_probabilities"])
            assert row["l0_batch_top1_expert"] == row["l1_batch_top1_expert"]
    for name in (
        "learning_curves.png",
        "regime_reconstruction.png",
        "expert_timeline.png",
        "paired_endpoints.png",
        "seed_summary.csv",
    ):
        assert (tmp_path / "analysis" / name).is_file()
    path = tmp_path / "seed_0/gmm/training.jsonl"
    rows = read_rows(path)
    rows[0]["mask_noise_hash"] = "wrong"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="masking differs"):
        analyze(tmp_path)


def test_forgetting_uses_post_block_baselines_and_pass_endpoints() -> None:
    """Do not conflate initial learning gains with subsequent deterioration."""
    rows = [{"step": step, "regime_mse": {"0": value}} for step, value in ((0, 2.0), (2, 1.0), (3, 0.8), (6, 1.4))]
    boundaries = [{"epoch": 0, "regime": 0, "exit_step": 2}]
    result = forgetting_summary(rows, boundaries, 6)[0]
    assert result["end_minus_exit"] == pytest.approx(0.4)
    assert result["end_minus_best_after_exit"] == pytest.approx(0.6)
    assert result["worst_minus_exit"] == pytest.approx(0.4)


@pytest.mark.parametrize("cap", [1.0, None])
@pytest.mark.parametrize("arm", ["learned", "gmm"])
def test_expert_learning_rates_scale_actual_adamw_updates_and_keep_moments(arm: str, cap: float | None) -> None:
    """Exercise post-normalization scaling, capped rates, shared parameters, and zero-rate momentum."""
    from cafl4ds.expert_learning_rates import (  # noqa: PLC0415
        apply_expert_learning_rates,
        expert_parameter_groups,
        expert_update_norms,
        snapshot_experts,
    )

    cfg = _config()
    ordinary, _ = build_method(cfg, 0, arm)
    scaled, _ = build_method(cfg, 0, arm)
    groups = expert_parameter_groups(scaled, 1e-3)
    grouped = [parameter for group in groups for parameter in group["params"]]
    assert len(grouped) == len({id(parameter) for parameter in grouped}) == len(list(scaled.parameters()))
    reference = torch.optim.AdamW(ordinary.parameters(), lr=1e-3, weight_decay=0.05)
    optimizer = torch.optim.AdamW(groups, lr=1e-3, weight_decay=0.05)
    masses = [0.6, 0.25, 0.01, 0.14]
    scales = [1, 1, 0.04, 0.56] if cap else [2.4, 1, 0.04, 0.56]
    for step in range(3):
        # Identical artificial gradients isolate optimization from changing forward representations.
        for left, right in zip(ordinary.parameters(), scaled.parameters(), strict=True):
            left.grad = torch.full_like(left, 0.2 if step < 2 else 0)
            right.grad = left.grad.clone()
        active = [0.25] * 4 if step == 0 else masses if step == 1 else [0, 0, 0, 1.0]
        readout = {f"l{layer}_mean_probabilities": active for layer in (0, 1)}
        logged = apply_expert_learning_rates(optimizer, readout, cap=cap)
        assert logged["shared_and_router_lr"] == 1e-3
        if step == 1:
            np.testing.assert_allclose(logged["l0_expert_lr_scale"], scales)
        before = {name: value.detach().clone() for name, value in scaled.named_parameters()}
        old_reference = {name: value.detach().clone() for name, value in ordinary.named_parameters()}
        snapshots = snapshot_experts(optimizer)
        reference.step()
        optimizer.step()
        changes = expert_update_norms(optimizer, snapshots)
        assert all(np.isfinite(value) and value >= 0 for values in changes.values() for value in values)
        for (name, left), right in zip(ordinary.named_parameters(), scaled.parameters(), strict=True):
            state_left, state_right = reference.state[left], optimizer.state[right]
            assert int(state_right["step"]) == step + 1
            torch.testing.assert_close(state_left["exp_avg"], state_right["exp_avg"], rtol=0, atol=0)
            torch.testing.assert_close(state_left["exp_avg_sq"], state_right["exp_avg_sq"], rtol=0, atol=0)
            if step == 0:
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            elif step == 1:
                scale = next(
                    group["lr"] / 1e-3
                    for group in optimizer.param_groups
                    if any(parameter is right for parameter in group["params"])
                )
                torch.testing.assert_close(
                    right - before[name], scale * (left - old_reference[name]), rtol=2e-4, atol=3e-8
                )
            elif any(
                parameter is right
                for group in optimizer.param_groups
                if group["lr"] == 0
                for parameter in group["params"]
            ):
                torch.testing.assert_close(right, before[name], rtol=0, atol=0)
                assert torch.count_nonzero(state_right["exp_avg"]) > 0


def test_expert_lr_ensemble_matches_original_and_audits_all_updates(tmp_path: Path) -> None:
    """Compare complete tiny ensembles and detect a corrupted expert-rate record."""
    from cafl4ds.expert_lr_analysis import audit_learning_rates, validate_baseline  # noqa: PLC0415

    cfg = _config()
    baseline = tmp_path / "baseline"
    run_experiment(cfg, baseline)
    cfg.experiment_id = "M6.2.0"
    cfg.expert_lr_scaling = True
    cfg.baseline_directory = str(baseline)
    output = tmp_path / "scaled"
    run_experiment(cfg, output)
    comparison = json.loads((output / "baseline_comparison.json").read_text())
    assert comparison["verification"]["matched_every_cross_experiment_initialization_order_view_and_mask"]
    assert comparison["all_saved_initial_tensors_match_baseline_exactly"]
    assert (output / "expert_displacement.json").is_file()
    assert comparison["learning_rate_audit"]["audited_gradient_steps"] == 24
    assert comparison["learning_rate_audit"]["audited_layer_lr_vectors"] == 48
    protocol = json.loads((output / "protocol.json").read_text())
    protocol["baseline_reference"]["comparison_sha256"] = "changed"
    with pytest.raises(ValueError, match="frozen baseline"):
        validate_baseline(protocol, baseline)
    path = output / "seed_0/gmm/training.jsonl"
    rows = read_rows(path)
    rows[0]["l0_expert_lr"][0] = 99
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(AssertionError):
        audit_learning_rates(output)


def test_additional_uncapped_gmm_arm_and_matched_analysis(tmp_path: Path) -> None:
    """Run only the added arm, preserve original results, and reject altered fixed routing."""
    from cafl4ds.gmm_uncapped_analysis import analyze_uncapped  # noqa: PLC0415

    cfg = _config()
    cfg.expert_lr_scaling = True
    baseline = tmp_path / "capped"
    run_experiment(cfg, baseline)
    original = (baseline / "comparison.json").read_bytes()
    OmegaConf.update(cfg, "arms", ["gmm"], force_add=True)
    OmegaConf.update(cfg, "expert_lr_cap", None, force_add=True)
    cfg.baseline_directory = str(baseline)
    output = tmp_path / "uncapped"
    result = run_experiment(cfg, output)
    assert result["verification"]["audited_added_gradient_steps"] == 12
    assert result["learning_rate_audit"]["all_expert_rates_match_4q"]
    assert set(result["conditions"]) == {"learned_capped", "gmm_capped", "gmm_uncapped"}
    assert not list(output.glob("seed_*/learned"))
    assert (baseline / "comparison.json").read_bytes() == original
    assert (output / "analysis/three_arm_learning.png").is_file()
    rows = read_rows(output / "seed_0/gmm/training.jsonl")
    assert any(max(row["l0_expert_lr_scale"]) > 1 for row in rows)
    for row in rows:
        np.testing.assert_allclose(row["l0_expert_lr_scale"], 4 * np.array(row["l0_mean_probabilities"]))
    rows[0]["l0_batch_top1_expert"] = -1
    path = output / "seed_0/gmm/training.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="Fixed routing differs"):
        analyze_uncapped(output, baseline)
