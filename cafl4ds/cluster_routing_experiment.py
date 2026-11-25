"""M6.1.0: paired top-4 learned versus fixed GMM image routing for sequential MAE."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import pickle
import platform
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import sklearn
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms

from cafl4ds import measurements
from cafl4ds.expert_learning_rates import (
    apply_expert_learning_rates,
    expert_parameter_groups,
    expert_update_norms,
    snapshot_experts,
)
from cafl4ds.models.fixed_mixture import FixedImageMixture, image_routing, replace_routers
from cafl4ds.models.vit import FeedForwardOutput, MoEMlp, MoEViTEncoder
from cafl4ds.routing_experiment import (
    Corpus,
    _atomic_checkpoint,
    _gradient_norm,
    digest_file,
    isolated_rng,
    keyed_seed,
    paired_views,
    reserve_panels,
    state_hash,
    stream_epoch_order,
    synchronize,
    tensor_hash,
    write_json,
    write_stream_manifest,
)
from cafl4ds.ssl.mae import MAE

ARMS = ("learned", "gmm")


@dataclass
class RegimePanel:
    """Fixed, label-stratified validation images; labels never enter a training forward."""

    images: torch.Tensor
    probabilities: torch.Tensor
    ids: torch.Tensor
    regimes: torch.Tensor
    names: list[str]


@dataclass
class ExperimentData:
    """Shared source, frozen image conditioning, reservations and independent regime panels."""

    corpus: Corpus
    probabilities: torch.Tensor
    panels: dict[str, list[int]]
    regime_panel: RegimePanel
    provenance: dict[str, Any]


def load_data(cfg: DictConfig) -> ExperimentData:
    """Join GMM rows to stable BDD image IDs and reserve independent official-val panels."""
    corpus = cast(Corpus, instantiate(cfg.source))
    panels = reserve_panels(corpus, cfg)
    if corpus.metadata.get("synthetic"):
        generator = torch.Generator().manual_seed(731)
        probabilities = torch.randn(len(corpus.images), 4, generator=generator).softmax(-1)
        ids = torch.tensor(panels["query"] + panels["natural"])
        regime_panel = RegimePanel(
            corpus.images[ids], probabilities[ids], ids, corpus.regimes[ids], [corpus.names[i] for i in ids]
        )
        return ExperimentData(corpus, probabilities, panels, regime_panel, {"synthetic": True})

    gmm_dir = Path(cfg.gmm_directory).resolve()
    manifest = json.loads((gmm_dir / "manifest.json").read_text())
    rows = {row["path"]: i for i, row in enumerate(manifest["images"])}
    if len(rows) != len(manifest["images"]):
        raise ValueError("GMM manifest contains duplicate image paths.")
    indices = np.array([rows[f"images/100k/train/{name}"] for name in corpus.names])
    all_probabilities = np.load(gmm_dir / "probabilities.npy", allow_pickle=False)
    if all_probabilities.shape != (len(rows), 4) or not np.isfinite(all_probabilities).all():
        raise ValueError("Expected complete, finite four-component GMM probabilities.")
    if (all_probabilities < 0).any() or not np.allclose(all_probabilities.sum(-1), 1, atol=1e-5):
        raise ValueError("GMM probabilities must be nonnegative and normalized.")
    # Verify the probability cache against the actual saved inference pipeline, not just filenames.
    with (gmm_dir / "pca_gmm.pkl").open("rb") as stream:
        pipeline = pickle.load(stream)  # noqa: S301 - trusted local M6.0.0 experiment artifact
    features = np.load(gmm_dir / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    np.testing.assert_allclose(pipeline.predict_proba(features[indices]), all_probabilities[indices], atol=3e-5)
    probabilities = torch.from_numpy(all_probabilities[indices].copy())
    probabilities /= probabilities.sum(-1, keepdim=True)

    bdd_root = Path(str(cfg.bdd_root or corpus.metadata["bdd_root"]))
    name_to_regime = {name: regime for regime, name in corpus.regime_names.items()}
    candidates: dict[int, list[str]] = {regime: [] for regime in corpus.regime_order}
    for label_path in sorted((bdd_root / "labels/100k/val").glob("*.json")):
        record = json.loads(label_path.read_text())
        attributes = record.get("attributes", {})
        regime = name_to_regime.get(f"{attributes.get('timeofday')}·{attributes.get('weather')}")
        if regime is not None:
            name = record["name"]
            candidates[regime].append(name if Path(name).suffix else name + ".jpg")
    generator = torch.Generator().manual_seed(int(cfg.split_seed))
    names: list[str] = []
    regimes: list[int] = []
    gmm_rows: list[int] = []
    for regime in corpus.regime_order:
        options = candidates[regime]
        if not options:
            raise ValueError(f"No independent validation image for {corpus.regime_names[regime]}.")
        order = torch.randperm(len(options), generator=generator)[: int(cfg.regime_eval_per)]
        chosen = [options[i] for i in order]
        names.extend(chosen)
        regimes.extend([regime] * len(chosen))
        gmm_rows.extend(rows[f"images/100k/val/{name}"] for name in chosen)
    transform = transforms.Compose([transforms.Resize((int(cfg.img_size), int(cfg.img_size))), transforms.ToTensor()])
    images = []
    for name in names:
        with Image.open(bdd_root / "images/100k/val" / name) as image:
            images.append(transform(image.convert("RGB")))
    val_probabilities = torch.from_numpy(all_probabilities[gmm_rows].copy())
    val_probabilities /= val_probabilities.sum(-1, keepdim=True)
    np.testing.assert_allclose(pipeline.predict_proba(features[gmm_rows]), all_probabilities[gmm_rows], atol=3e-5)
    regime_panel = RegimePanel(
        torch.stack(images), val_probabilities, torch.tensor(gmm_rows), torch.tensor(regimes), names
    )
    provenance = {
        "gmm_directory": str(gmm_dir),
        "gmm_model_sha256": digest_file(gmm_dir / "pca_gmm.pkl"),
        "gmm_manifest_sha256": digest_file(gmm_dir / "manifest.json"),
        "gmm_probability_sha256": digest_file(gmm_dir / "probabilities.npy"),
        "training_gmm_row_indices": indices.tolist(),
        "training_probabilities_sha256": tensor_hash(probabilities),
        "validation_gmm_row_indices": gmm_rows,
        "validation_names": names,
        "validation_regimes": regimes,
        "validation_images_sha256": tensor_hash(regime_panel.images),
        "validation_probabilities_sha256": tensor_hash(val_probabilities),
        "regime_eval_counts": dict(Counter(regimes)),
        "gmm_scope": "M6.0.0 fitted all train/val/test images; deliberate offline look-ahead conditioning",
    }
    return ExperimentData(corpus, probabilities, panels, regime_panel, provenance)


def build_method(cfg: DictConfig, seed: int, arm: str) -> tuple[MAE, str]:
    """Use identical shared random initialization; fixed routing has no router parameters."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    with isolated_rng(keyed_seed("initialization", seed), str(cfg.device)):
        method = cast(MAE, instantiate(cfg.ssl, encoder=instantiate(cfg.encoder)))
    layers = [layer for layer in method.modules() if isinstance(layer, MoEMlp)]
    if not layers or any(
        layer.top_k != 4 or layer.num_experts != 4 or layer.routing_level != "image" for layer in layers
    ):
        raise ValueError("M6.1.0 requires four experts, top-4, image-level routing.")
    shared = {name: value for name, value in method.state_dict().items() if ".router." not in name}
    initial_hash = state_hash(shared)
    if arm == "gmm":
        replace_routers(method)
    return method.to(cfg.device), initial_hash


class StepRoutingReadout:
    """Record actual training gate weights, including router noise, from the update forward."""

    def __init__(self, method: MAE) -> None:
        """Observe each MoE block; top-1 refers to minibatch mean probabilities."""
        self.rows: dict[str, Any] = {}
        self.image_probabilities: dict[int, torch.Tensor] = {}
        self.handles = []
        for index, block in enumerate(method.encoder.blocks):
            if not isinstance(block.mlp, MoEMlp | FixedImageMixture):
                continue

            def capture(
                _module: torch.nn.Module, _args: tuple[torch.Tensor], result: FeedForwardOutput, layer: int = index
            ) -> None:
                if result.routing_probs is None:
                    raise RuntimeError("Missing probabilities in the actual update forward.")
                p = result.routing_probs.detach()[:, 0].float()
                self.image_probabilities[layer] = p.cpu()
                mass = p.mean(0).cpu()
                counts = torch.bincount(p.argmax(-1), minlength=p.shape[-1]).cpu()
                self.rows.update(
                    {
                        f"l{layer}_batch_top1_expert": int(mass.argmax()),
                        f"l{layer}_mean_probabilities": mass.tolist(),
                        f"l{layer}_image_top1_counts": counts.tolist(),
                        f"l{layer}_entropy": float(
                            -(p * p.clamp_min(1e-30).log()).sum(-1).mean() / math.log(p.shape[-1])
                        ),
                    }
                )

            self.handles.append(block.mlp.register_forward_hook(capture))

    def close(self) -> None:
        """Release observers before evaluation or after errors."""
        for handle in self.handles:
            handle.remove()


def _embeddings(method: MAE, images: torch.Tensor, probabilities: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
    batches = []
    for start in range(0, len(images), batch_size):
        with image_routing(method, probabilities[start : start + batch_size]):
            batches.append(method.encode(images[start : start + batch_size]).cpu())
    return torch.cat(batches)


def _reconstruction(
    method: MAE,
    images: torch.Tensor,
    probabilities: torch.Tensor,
    ids: torch.Tensor,
    cfg: DictConfig,
) -> torch.Tensor:
    draws = []
    for draw in range(int(cfg.mask_draws)):
        batches = []
        for start in range(0, len(images), int(cfg.batch_size)):
            stop = start + int(cfg.batch_size)
            views, noise = paired_views(method, images[start:stop], ids[start:stop], 910_000, draw)
            with image_routing(method, probabilities[start:stop]):
                losses, mask, _ = method._masked_recon_loss(views.to(cfg.device), augmented=True, mask_noise=noise)
            batches.append(((losses * mask).sum(-1) / mask.sum(-1)).cpu())
        draws.append(torch.cat(batches))
    return torch.stack(draws).mean(0)


@torch.no_grad()  # type: ignore[misc]
def evaluate(
    method: MAE,
    data: ExperimentData,
    cfg: DictConfig,
    step: int,
    reference: torch.Tensor | None,
    initial_routes: dict[int, torch.Tensor] | None,
    previous_routes: dict[int, torch.Tensor] | None,
    probes: bool,
) -> tuple[dict[str, Any], torch.Tensor, dict[int, torch.Tensor], dict[str, np.ndarray]]:
    """Measure the standard health suite and all regimes using fixed views/masks and IDs."""
    modes = [(module, module.training) for module in method.modules()]
    method.eval()
    try:
        with isolated_rng(900_000, str(cfg.device)):
            return _evaluate(method, data, cfg, step, reference, initial_routes, previous_routes, probes)
    finally:
        for module, training in modes:
            module.training = training


def _evaluate(
    method: MAE,
    data: ExperimentData,
    cfg: DictConfig,
    step: int,
    reference: torch.Tensor | None,
    initial_routes: dict[int, torch.Tensor] | None,
    previous_routes: dict[int, torch.Tensor] | None,
    probes: bool,
) -> tuple[dict[str, Any], torch.Tensor, dict[int, torch.Tensor], dict[str, np.ndarray]]:
    corpus, panels, p = data.corpus, data.panels, data.probabilities
    query = torch.tensor(panels["query"])
    natural = torch.tensor(panels["natural"])
    features = _embeddings(method, corpus.images[query], p[query])
    unit = torch.nn.functional.normalize(features, dim=-1)
    routing_batches = []
    for start in range(0, len(query), 64):
        ids = query[start : start + 64]
        with image_routing(method, p[ids]):
            routing_batches.append(cast(MoEViTEncoder, method.encoder).routing(corpus.images[ids]))
    routes = {layer: torch.cat([batch[layer][:, 0].cpu() for batch in routing_batches]) for layer in routing_batches[0]}
    natural_errors = _reconstruction(method, corpus.images[natural], p[natural], natural, cfg)
    balanced_errors = _reconstruction(method, corpus.images[query], p[query], query, cfg)
    panel = data.regime_panel
    regime_errors = _reconstruction(method, panel.images, panel.probabilities, panel.ids, cfg)
    regimes = {str(regime): float(regime_errors[panel.regimes == regime].mean()) for regime in corpus.regime_order}
    metrics: dict[str, Any] = {
        "step": step,
        "heldout_mse": float(natural_errors.mean()),
        "balanced_mse": float(balanced_errors.mean()),
        "regime_mse": regimes,
        "rankme": measurements.rankme(features),
        "mean_feature_var": float(measurements.feature_variance(unit).mean()),
        "offdiag_cov": measurements.offdiag_covariance(unit),
        "uniformity": measurements.uniformity(features),
        "cka_drift": 0.0 if reference is None else measurements.cka_drift(reference, features),
        "cosine_drift": 0.0 if reference is None else measurements.cosine_drift(reference, features),
    }
    first, _ = paired_views(method, corpus.images[query], query, 812_000, 0)
    second, _ = paired_views(method, corpus.images[query], query, 812_000, 1)
    metrics["alignment"] = measurements.alignment(
        _embeddings(method, first, p[query]), _embeddings(method, second, p[query])
    )
    entropies = []
    for layer, values in routes.items():
        entropy = -(values * values.clamp_min(1e-30).log()).sum(-1) / math.log(values.shape[-1])
        entropies.append(entropy)
        counts = torch.bincount(values.argmax(-1), minlength=values.shape[-1]).float()
        load = counts / counts.sum()
        metrics.update(
            {
                f"l{layer}_entropy": float(entropy.mean()),
                f"l{layer}_mean_probabilities": values.mean(0).tolist(),
                f"l{layer}_image_top1_fraction": load.tolist(),
                f"l{layer}_soft_mass_cv": float(values.mean(0).std(unbiased=False) / values.mean()),
                f"l{layer}_route_change_initial": 0.0
                if initial_routes is None
                else float((values.argmax(-1) != initial_routes[layer].argmax(-1)).float().mean()),
                f"l{layer}_route_change_previous": 0.0
                if previous_routes is None
                else float((values.argmax(-1) != previous_routes[layer].argmax(-1)).float().mean()),
                f"l{layer}_argmax_unrepresented_fraction": float((counts == 0).float().mean()),
            }
        )
    metrics["entropy_mean"] = float(torch.stack(entropies).mean())
    if probes:
        support = torch.tensor(panels["support"])
        support_features = _embeddings(method, corpus.images[support], p[support])
        natural_features = _embeddings(method, corpus.images[natural], p[natural])
        for name, queries, ids in (("balanced", features, query), ("natural", natural_features, natural)):
            source = (support_features, corpus.labels[support])
            target = (queries, corpus.labels[ids])
            metrics[name + "_knn"] = measurements.knn_probe(lambda x: x, source, target, k=20)
            metrics[name + "_linear"] = measurements.linear_probe(lambda x: x, source, target)
    arrays = {
        "features": features.numpy(),
        "natural_reconstruction": natural_errors.numpy(),
        "regime_reconstruction": regime_errors.numpy(),
        **{f"l{layer}_probabilities": values.numpy() for layer, values in routes.items()},
    }
    return metrics, features, routes, arrays


def block_boundaries(data: ExperimentData, cfg: DictConfig, seed: int) -> list[dict[str, Any]]:
    """Map exact regime exits to the first completed update containing the exit image."""
    boundaries = []
    steps_per_epoch = len(data.panels["train"]) // int(cfg.batch_size)
    for epoch in range(int(cfg.epochs)):
        order = stream_epoch_order(data.panels["train"], seed, epoch, data.corpus, cfg)
        labels = data.corpus.regimes[order[: steps_per_epoch * int(cfg.batch_size)]]
        for regime in data.corpus.regime_order:
            positions = (labels == regime).nonzero().flatten()
            if len(positions):
                boundaries.append(
                    {
                        "epoch": epoch,
                        "regime": regime,
                        "start_step": epoch * steps_per_epoch + float(positions[0]) / int(cfg.batch_size),
                        "exit_step": epoch * steps_per_epoch
                        + math.ceil((int(positions[-1]) + 1) / int(cfg.batch_size)),
                        "count": len(positions),
                    }
                )
    return boundaries


def forgetting_summary(
    rows: list[dict[str, Any]], boundaries: list[dict[str, Any]], steps_per_epoch: int
) -> list[dict[str, Any]]:
    """Report signed endpoint deterioration and nonnegative forgetting, separately per pass."""
    result = []
    by_step = {row["step"]: row for row in rows}
    for boundary in boundaries:
        start = boundary["exit_step"]
        end = (boundary["epoch"] + 1) * steps_per_epoch
        if start not in by_step or end not in by_step:
            continue
        regime = str(boundary["regime"])
        values = [row["regime_mse"][regime] for row in rows if start <= row["step"] <= end]
        baseline, final = by_step[start]["regime_mse"][regime], by_step[end]["regime_mse"][regime]
        result.append(
            {
                **boundary,
                "pass_end_step": end,
                "post_exit_updates": end - start,
                "mse_at_exit": baseline,
                "mse_at_pass_end": final,
                "end_minus_exit": final - baseline,
                "end_minus_best_after_exit": final - min(values),
                "worst_minus_exit": max(values) - baseline,
            }
        )
    return result


def run_arm(  # noqa: C901 - one transaction owns step-keyed RNG, logs, and checkpoint offsets
    cfg: DictConfig,
    data: ExperimentData,
    seed: int,
    arm: str,
    directory: Path,
    protocol_hash: str,
    stop_after: int | None = None,
) -> dict[str, Any]:
    """Train or exactly resume a complete paired arm; all incoming training images are used."""
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "complete.json").exists():
        complete: dict[str, Any] = json.loads((directory / "complete.json").read_text())
        if complete["protocol_hash"] != protocol_hash:
            raise ValueError("Completed result belongs to a different protocol.")
        return complete
    device = str(cfg.device)
    method, initial_hash = build_method(cfg, seed, arm)
    scaled_lr = bool(cfg.get("expert_lr_scaling", False))
    parameters = expert_parameter_groups(method, float(cfg.optim.lr)) if scaled_lr else method.parameters()
    optimizer = instantiate(cfg.optim, _partial_=True)(params=parameters)
    per_epoch = len(data.panels["train"]) // int(cfg.batch_size)
    total = min(int(cfg.epochs) * per_epoch, int(cfg.max_steps or int(cfg.epochs) * per_epoch))
    boundaries = block_boundaries(data, cfg, seed)
    exit_steps = {row["exit_step"] for row in boundaries}
    probe_steps = {0, total, round(total / 4), round(total / 2)}
    paths = {name: directory / f"{name}.jsonl" for name in ("training", "evaluation")}
    checkpoint = directory / "latest.pt"
    step, reference, initial_routes, previous = 0, None, None, None
    elapsed = {"preparation": 0.0, "update": 0.0, "evaluation": 0.0}
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state["protocol_hash"] != protocol_hash or state["shared_initial_hash"] != initial_hash:
            raise ValueError("Checkpoint data, code, configuration or shared initialization changed.")
        method.load_state_dict(state["method"])
        optimizer.load_state_dict(state["optimizer"])
        step, reference = state["step"], state["reference"].cpu()
        initial_routes = {key: value.cpu() for key, value in state["initial_routes"].items()}
        previous = {key: value.cpu() for key, value in state["previous"].items()}
        elapsed = state["elapsed"]
        for name, path in paths.items():
            with path.open("r+b") as stream:
                stream.truncate(state["log_offsets"][name])
    logs = {name: path.open("a") for name, path in paths.items()}

    def append(name: str, row: dict[str, Any]) -> None:
        logs[name].write(json.dumps(row, allow_nan=False) + "\n")
        logs[name].flush()

    def save() -> None:
        _atomic_checkpoint(
            checkpoint,
            {
                "protocol_hash": protocol_hash,
                "shared_initial_hash": initial_hash,
                "method": method.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "reference": reference,
                "initial_routes": initial_routes,
                "previous": previous,
                "elapsed": elapsed,
                "log_offsets": {name: stream.tell() for name, stream in logs.items()},
            },
        )

    def measure() -> None:
        nonlocal reference, initial_routes, previous
        synchronize(device)
        started = time.perf_counter()
        metrics, features, routes, arrays = evaluate(
            method, data, cfg, step, reference, initial_routes, previous, step in probe_steps
        )
        if reference is None:
            reference = features
            initial_routes = routes
        previous = routes
        synchronize(device)
        elapsed["evaluation"] += time.perf_counter() - started
        append("evaluation", metrics)
        np.savez_compressed(directory / f"panel_{step:06d}.npz", **cast(dict[str, Any], arrays))
        logger.info(
            "seed={} arm={} step={}/{} mse={:.5f} entropy={:.3f}",
            seed,
            arm,
            step,
            total,
            metrics["heldout_mse"],
            metrics["entropy_mean"],
        )

    try:
        if not checkpoint.exists():
            measure()
            save()
            shutil.copyfile(checkpoint, directory / "initial.pt")
        current_epoch, order = -1, torch.empty(0, dtype=torch.long)
        while step < total:
            epoch, offset = divmod(step, per_epoch)
            if epoch != current_epoch:
                order = stream_epoch_order(data.panels["train"], seed, epoch, data.corpus, cfg)
                current_epoch = epoch
            ids = order[offset * int(cfg.batch_size) : (offset + 1) * int(cfg.batch_size)]
            started = time.perf_counter()
            views, noise = paired_views(method, data.corpus.images[ids], ids, seed, epoch)
            elapsed["preparation"] += time.perf_counter() - started
            method.train()
            optimizer.zero_grad(set_to_none=True)
            observer = StepRoutingReadout(method)
            synchronize(device)
            started = time.perf_counter()
            try:
                with (
                    isolated_rng(keyed_seed("router_noise", seed, step), device),
                    image_routing(method, data.probabilities[ids]),
                ):
                    losses = method.loss_components(views.to(device), augmented=True, mask_noise=noise)
                    losses.total.backward()
                gradients: dict[str, Any] = {}
                for layer, block in enumerate(method.encoder.blocks):
                    if isinstance(block.mlp, MoEMlp | FixedImageMixture):
                        gradients[f"l{layer}_expert_grad_norm"] = [
                            _gradient_norm(e.parameters()) for e in block.mlp.experts
                        ]
                        gradients[f"l{layer}_router_grad_norm"] = (
                            _gradient_norm(block.mlp.router.parameters()) if isinstance(block.mlp, MoEMlp) else None
                        )
                grad_norm = _gradient_norm(method.parameters())
                if not math.isfinite(float(losses.total.detach())) or not math.isfinite(grad_norm):
                    raise FloatingPointError("Non-finite loss/gradient; refusing optimizer update.")
                torch.nn.utils.clip_grad_norm_(method.parameters(), float(cfg.grad_clip))
                learning_rates = (
                    apply_expert_learning_rates(optimizer, observer.rows, cap=cfg.get("expert_lr_cap", 1.0))
                    if scaled_lr
                    else {}
                )
                before = snapshot_experts(optimizer) if scaled_lr else {}
                optimizer.step()
                updates = expert_update_norms(optimizer, before) if scaled_lr else {}
                synchronize(device)
            finally:
                observer.close()
            elapsed["update"] += time.perf_counter() - started
            step += 1
            regime_counts = torch.bincount(data.corpus.regimes[ids], minlength=len(data.corpus.regime_order))
            regime_mass = {
                f"l{layer}_mass_by_regime": {
                    str(regime): values[data.corpus.regimes[ids] == regime].sum(0).tolist()
                    for regime in data.corpus.regimes[ids].unique().tolist()
                }
                for layer, values in observer.image_probabilities.items()
            }
            append(
                "training",
                {
                    "step": step,
                    "epoch": epoch,
                    "sample_ids": ids.tolist(),
                    "order_hash": tensor_hash(ids),
                    "views_hash": tensor_hash(views),
                    "mask_noise_hash": tensor_hash(noise),
                    "regime_counts": regime_counts.tolist(),
                    "dominant_regime": int(regime_counts.argmax()),
                    "n_images": len(ids),
                    "reconstruction": float(losses.reconstruction.detach()),
                    "auxiliary": float(losses.auxiliary.detach()),
                    "weighted_auxiliary": float(losses.auxiliary.detach()) * method.aux_weight,
                    "loss": float(losses.total.detach()),
                    "grad_norm": grad_norm,
                    **observer.rows,
                    **gradients,
                    **regime_mass,
                    **learning_rates,
                    **updates,
                },
            )
            if step % int(cfg.eval_every) == 0 or step in exit_steps | probe_steps or step == stop_after:
                measure()
                save()
                if step in probe_steps:
                    shutil.copyfile(checkpoint, directory / f"checkpoint_{step:06d}.pt")
            if step == stop_after:
                return {"interrupted": True, "step": step}
        rows = [json.loads(line) for line in paths["evaluation"].read_text().splitlines()]
        forgetting = forgetting_summary(rows, boundaries, per_epoch)
        result = {
            "protocol_hash": protocol_hash,
            "seed": seed,
            "arm": arm,
            "shared_initial_hash": initial_hash,
            "steps": step,
            "steps_per_epoch": per_epoch,
            "training_exposures": step * int(cfg.batch_size),
            "initial": rows[0],
            "final": rows[-1],
            "forgetting": forgetting,
            "elapsed_seconds": elapsed,
            "parameters": sum(parameter.numel() for parameter in method.parameters()),
            "trainable_router_parameters": sum(p.numel() for n, p in method.named_parameters() if ".router." in n),
            "expert_lr_scaling": scaled_lr,
        }
        write_json(directory / "complete.json", result)
        return result
    finally:
        for log_stream in logs.values():
            log_stream.close()


def _training_arms(cfg: DictConfig, seed: int) -> list[str]:
    arms = list(cfg.get("arms", ARMS))
    if arms not in [list(ARMS), ["gmm"]]:
        raise ValueError("Choose both original arms or the additional GMM-only arm.")
    return arms if seed % 2 == 0 else list(reversed(arms))


def _validate_config(cfg: DictConfig) -> None:
    _training_arms(cfg, 0)
    if min(int(cfg.epochs), int(cfg.batch_size), int(cfg.eval_every), int(cfg.mask_draws)) < 1:
        raise ValueError("Epochs, batch size, evaluation cadence and mask draws must be positive.")
    if str(cfg.device) == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this process.")
    if str(cfg.stream_order) != "regime_blocked" or len(set(cfg.seeds)) != len(cfg.seeds) or not cfg.seeds:
        raise ValueError("Require the regime-blocked stream and unique nonempty paired seeds.")
    if min(int(cfg.threads), int(cfg.get("workers", 1)), int(cfg.regime_eval_per)) < 1:
        raise ValueError("Thread, worker and regime panel counts must be positive.")
    if cfg.max_steps is not None and int(cfg.max_steps) < 1:
        raise ValueError("An optional step cap must be positive.")


def run_experiment(cfg: DictConfig, output: Path) -> dict[str, Any]:
    """Freeze source/data provenance, execute the paired ensemble and verify its pairing."""
    _validate_config(cfg)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(cfg.threads))
    data = load_data(cfg)
    if tuple(data.corpus.images.shape[-2:]) != (int(cfg.img_size), int(cfg.img_size)):
        raise ValueError("Model resolution differs from the shared image cache.")
    files = [
        Path(__file__),
        Path(__file__).with_name("routing_experiment.py"),
        Path(__file__).with_name("routing_streams.py"),
        Path(__file__).with_name("measurements.py"),
        Path(__file__).with_name("expert_learning_rates.py"),
        *Path(__file__).parent.joinpath("models").glob("*.py"),
        *Path(__file__).parent.joinpath("ssl").glob("*.py"),
    ]
    protocol = {
        "experiment": str(cfg.get("experiment_id", "M6.1.0")),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "source": data.corpus.fingerprint,
        "conditioning": data.provenance,
        "panels": data.panels,
        "source_hashes": {str(path.relative_to(Path(__file__).parent)): digest_file(path) for path in files},
        "runtime": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "python": platform.python_version(),
        },
        "primary_outcomes": ["natural_heldout_mse", "per_pass_regime_endpoint_minus_exit_mse"],
        "pairing": "shared weights, IDs, order, augmentation and masks within seed; router noise learned arm only",
        "routing": "top-4 full mixture, one distribution per image per layer; no filtering or token dropping",
        "step_top1": "argmax of mean image probabilities in each minibatch, per MoE layer; all four experts execute",
        "regime_evaluation": ("independent official validation panels, up to 32 images/regime; fixed view/mask draws"),
        "expert_lr_rule": (
            (
                "lr_e = base_lr * E * mean actual minibatch probability_e, per layer, uncapped. "
                if cfg.get("expert_lr_cap", 1.0) is None
                else "lr_e = base_lr * min(1, E * mean actual minibatch probability_e), per layer. "
            )
            + "Shared/router lr stays at base_lr. AdamW moments, counters and clipping unchanged."
            if cfg.get("expert_lr_scaling", False)
            else "Constant AdamW learning rate for every parameter."
        ),
    }
    if cfg.get("baseline_directory"):
        from cafl4ds.expert_lr_analysis import validate_baseline  # noqa: PLC0415

        protocol["baseline_reference"] = validate_baseline(protocol, Path(cfg.baseline_directory).resolve())
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    path = output / "protocol.json"
    if path.exists() and json.loads(path.read_text())["protocol_hash"] != protocol_hash:
        raise ValueError("Refusing to mix different frozen protocols. Choose a fresh output directory.")
    write_json(path, {**protocol, "protocol_hash": protocol_hash})
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    write_stream_manifest(cfg, data.corpus, data.panels, output)
    write_json(
        output / "membership.json",
        {
            "names": data.corpus.names,
            "regimes": data.corpus.regimes.tolist(),
            "regime_names": data.corpus.regime_names,
            "scene_names": data.corpus.scene_names,
            "regime_eval_names": data.regime_panel.names,
            "regime_eval_regimes": data.regime_panel.regimes.tolist(),
        },
    )
    for source in files:
        destination = output / "provenance" / source.relative_to(Path(__file__).parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    results = []
    if int(cfg.get("workers", 1)) > 1:
        config = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
        with ProcessPoolExecutor(
            max_workers=int(cfg.workers), mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            futures = [executor.submit(_run_seed, config, output, int(seed), protocol_hash) for seed in cfg.seeds]
            for future in futures:
                results.extend(future.result())
    else:
        for seed in cfg.seeds:
            # Alternate execution order across paired seeds to limit systematic timing-order effects.
            for arm in _training_arms(cfg, int(seed)):
                results.append(run_arm(cfg, data, int(seed), arm, output / f"seed_{seed}" / arm, protocol_hash))
    return _analyze_results(cfg, output, results)


def _analyze_results(cfg: DictConfig, output: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    if list(cfg.get("arms", ARMS)) == ["gmm"]:
        from cafl4ds.gmm_uncapped_analysis import analyze_uncapped  # noqa: PLC0415

        return analyze_uncapped(output, Path(cfg.baseline_directory).resolve())
    from cafl4ds.cluster_routing_analysis import analyze  # noqa: PLC0415 - avoid a circular authoring dependency

    comparison = analyze(output, results)
    if cfg.get("baseline_directory"):
        from cafl4ds.expert_lr_analysis import compare_to_baseline  # noqa: PLC0415

        compare_to_baseline(output, Path(cfg.baseline_directory).resolve())
    return comparison


def _run_seed(config: dict[str, Any], output: Path, seed: int, protocol_hash: str) -> list[dict[str, Any]]:
    """Run an entire seed pair in its own process, loading the shared read-only image mmap."""
    cfg = OmegaConf.create(config)
    torch.set_num_threads(int(cfg.threads))
    protocol = json.loads((output / "protocol.json").read_text())
    for name, fingerprint in protocol["source_hashes"].items():
        if digest_file(Path(__file__).parent / name) != fingerprint:
            raise ValueError("Source changed after the experiment protocol was frozen.")
    data = load_data(cfg)
    if (
        data.corpus.fingerprint != protocol["source"]
        or json.loads(json.dumps(data.provenance)) != protocol["conditioning"]
        or data.panels != protocol["panels"]
    ):
        raise ValueError("Worker data or conditioning changed after protocol creation.")
    return [
        run_arm(cfg, data, seed, arm, output / f"seed_{seed}" / arm, protocol_hash) for arm in _training_arms(cfg, seed)
    ]
