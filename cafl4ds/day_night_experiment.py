"""M6.3.0 paired MAE training on one day+dusk pass followed by one night pass."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import shutil
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import nn

from cafl4ds.cluster_routing_experiment import ExperimentData, StepRoutingReadout, build_method, load_data
from cafl4ds.expert_learning_rates import apply_expert_learning_rates, expert_parameter_groups
from cafl4ds.models.fixed_mixture import image_routing
from cafl4ds.models.heads import MAEDecoder
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.routing_experiment import (
    _atomic_checkpoint,
    digest_file,
    epoch_order,
    isolated_rng,
    keyed_seed,
    paired_views,
    state_hash,
    synchronize,
    tensor_hash,
    write_json,
)
from cafl4ds.ssl.mae import MAE
from cafl4ds.ssl.trunk_mixture import TrunkMixtureMAE

ARMS = ("vmoe_top4", "vmoe_top1", "gmm_capped", "gmm_full", "gmm_split", "gmm_full_private_decoder")
TRUNK_ARMS = ARMS[3:]
PHASES = ("day_dusk", "night")


def _dense_encoder(cfg: DictConfig) -> TinyViTEncoder:
    return TinyViTEncoder(
        img_size=cfg.img_size,
        patch_size=cfg.encoder.patch_size,
        in_chans=cfg.encoder.in_chans,
        embed_dim=cfg.encoder.embed_dim,
        depth=cfg.encoder.depth,
        num_heads=cfg.encoder.num_heads,
        mlp_ratio=cfg.encoder.mlp_ratio,
    )


def build_arm(cfg: DictConfig, seed: int, arm: str) -> MAE:
    """Pair common initialization for the MoEs, private trunks, and shared decoders."""
    if arm not in TRUNK_ARMS:
        method, _ = build_method(cfg, seed, "gmm" if arm == "gmm_capped" else "learned")
        if arm == "vmoe_top1":
            for block in method.encoder.blocks:
                if hasattr(block.mlp, "top_k"):
                    block.mlp.top_k = 1
        return method

    # The common decoder and shared prefix start from the same tensors as the MoE controls.
    control, _ = build_method(cfg, seed, "gmm")
    encoders = []
    for expert in range(4):
        with isolated_rng(keyed_seed("m630_trunk", seed, expert), str(cfg.device)):
            encoders.append(_dense_encoder(cfg))
    shared_prefix = arm == "gmm_split"
    if shared_prefix:
        encoder = control.encoder
        encoder.blocks = nn.ModuleList(list(encoder.blocks)[:2])
        encoder.norm = nn.Identity()
        branches = [nn.Sequential(*list(trunk.blocks)[2:], trunk.norm) for trunk in encoders]
    else:
        encoder, branches = encoders[0], list(encoders[1:])
    decoders = []
    if arm == "gmm_full_private_decoder":
        for expert in range(1, 4):
            with isolated_rng(keyed_seed("m630_decoder", seed, expert), str(cfg.device)):
                decoders.append(
                    MAEDecoder(
                        num_patches=encoder.num_patches,
                        encoder_dim=encoder.embed_dim,
                        patch_size=encoder.patch_size,
                        in_chans=encoder.in_chans,
                        decoder_dim=cfg.ssl.decoder_dim,
                        depth=cfg.ssl.decoder_depth,
                        num_heads=cfg.ssl.decoder_heads,
                        mlp_ratio=cfg.ssl.decoder_mlp_ratio,
                    )
                )
    method = TrunkMixtureMAE(
        encoder,
        control.decoder,
        branches,
        decoders,
        shared_prefix=shared_prefix,
        mask_ratio=control.mask_ratio,
        norm_pix_loss=control.norm_pix_loss,
    )
    method.to(cfg.device)
    return method


def _trunk_groups(method: TrunkMixtureMAE, base_lr: float) -> list[dict[str, Any]]:
    groups = []
    expert_ids: set[int] = set()
    for expert, modules in enumerate(method.expert_modules()):
        parameters = [parameter for module in modules for parameter in module.parameters()]
        ids = {id(parameter) for parameter in parameters}
        if len(ids) != len(parameters) or expert_ids & ids:
            raise ValueError("A private parameter has multiple expert owners.")
        expert_ids.update(ids)
        groups.append(
            {
                "params": parameters,
                "name": f"l0_expert_{expert}",
                "base_lr": base_lr,
                "lr": base_lr,
                "expert_layer": 0,
                "expert_index": expert,
                "num_experts": 4,
            }
        )
    shared = [parameter for parameter in method.parameters() if id(parameter) not in expert_ids]
    if shared:
        groups.insert(0, {"params": shared, "base_lr": base_lr, "lr": base_lr, "name": "shared"})
    return groups


def optimizer_groups(method: MAE, arm: str, base_lr: float) -> Iterable[Any]:
    """Keep the earlier MLP rate rule and extend it to each private trunk and decoder."""
    if isinstance(method, TrunkMixtureMAE):
        return _trunk_groups(method, base_lr)
    if arm == "gmm_capped":
        return expert_parameter_groups(method, base_lr)
    return cast(Iterable[torch.Tensor], method.parameters())


def _time_labels(regimes: torch.Tensor, names: dict[int, str], synthetic: bool) -> torch.Tensor:
    if synthetic:
        return 2 * (regimes == max(names)).long()
    codes = {"daytime": 0, "dawn/dusk": 1, "night": 2}
    labels = torch.full_like(regimes, -1)
    for regime, name in names.items():
        labels[regimes == regime] = codes[name.split("·")[0]]
    if bool((labels < 0).any()):
        raise ValueError("Unrecognized time-of-day labels in the existing cache.")
    return labels


def phase_orders(data: ExperimentData, cfg: DictConfig, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Shuffle inside each phase and retain all images, including each short final batch."""
    labels = _time_labels(data.corpus.regimes, data.corpus.regime_names, bool(data.provenance.get("synthetic")))
    orders = [
        epoch_order(selected.nonzero().flatten().tolist(), seed, phase)
        for phase, selected in enumerate((labels < 2, labels == 2))
    ]
    if cfg.pilot_images_per_phase is not None:
        orders = [order[: int(cfg.pilot_images_per_phase)] for order in orders]
    if not all(len(order) for order in orders):
        raise ValueError("Both day+dusk and night need training examples.")
    return orders[0], orders[1]


@dataclass
class PreparedPanel:
    """Fixed validation views, masks, and probabilities shared by each worker's runs."""

    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    labels: torch.Tensor
    draws: int
    fingerprint: str


def prepare_panel(cfg: DictConfig, data: ExperimentData) -> PreparedPanel:
    """Precompute the same four image-keyed masks and views used at every evaluation."""
    panel = data.regime_panel
    labels = _time_labels(panel.regimes, data.corpus.regime_names, bool(data.provenance.get("synthetic")))
    method, _ = build_method(cfg, 0, "learned")
    batches = []
    hashes: list[str] = []
    for draw in range(int(cfg.mask_draws)):
        for start in range(0, len(panel.ids), int(cfg.batch_size)):
            stop = start + int(cfg.batch_size)
            views, noise = paired_views(method, panel.images[start:stop], panel.ids[start:stop], 910_000, draw)
            hashes.extend((tensor_hash(views), tensor_hash(noise)))
            batches.append((views, noise, panel.probabilities[start:stop]))
    return PreparedPanel(batches, labels, int(cfg.mask_draws), hashlib.sha256("".join(hashes).encode()).hexdigest())


def _conditioning(method: MAE, probabilities: torch.Tensor) -> AbstractContextManager[None]:
    return (
        method.image_routing(probabilities)
        if isinstance(method, TrunkMixtureMAE)
        else image_routing(method, probabilities)
    )


@torch.no_grad()  # type: ignore[misc]
def evaluate(method: MAE, panel: PreparedPanel, device: str) -> dict[str, float]:
    """Read masked-patch MSE with fixed views and masks and deterministic routing."""
    method.eval()
    errors = []
    for views, noise, probabilities in panel.batches:
        with _conditioning(method, probabilities):
            loss, mask, _ = method._masked_recon_loss(views.to(device), augmented=True, mask_noise=noise)
        errors.append(((loss * mask).sum(dim=-1) / mask.sum(dim=-1)).cpu())
    per_image = torch.cat(errors).reshape(panel.draws, len(panel.labels)).mean(0)
    subsets = {
        "day_dusk": panel.labels < 2,
        "day": panel.labels == 0,
        "dusk": panel.labels == 1,
        "night": panel.labels == 2,
    }
    return {f"{name}_mse": float(per_image[selected].mean()) for name, selected in subsets.items() if selected.any()}


def forward_gflops(cfg: DictConfig, arm: str) -> float:
    """Estimate forward matmul FLOPs per training image including patch projection and decoder."""
    patches = (cfg.img_size // cfg.encoder.patch_size) ** 2
    visible = 1 + max(1, round(patches * (1 - cfg.ssl.mask_ratio)))
    width, decoder_width = cfg.encoder.embed_dim, cfg.ssl.decoder_dim

    def block(tokens: int, dim: int, hidden: int, active_mlps: int = 1) -> int:
        return 2 * tokens * (4 * dim**2 + 2 * dim * hidden * active_mlps) + 4 * tokens**2 * dim

    hidden = int(width * cfg.encoder.mlp_ratio)
    dense = block(visible, width, hidden)
    depth = int(cfg.encoder.depth)
    patch_projection = 2 * patches * width * cfg.encoder.patch_size**2 * cfg.encoder.in_chans
    decoder = (
        2 * visible * width * decoder_width
        + cfg.ssl.decoder_depth * block(patches + 1, decoder_width, int(decoder_width * cfg.ssl.decoder_mlp_ratio))
        + 2 * patches * decoder_width * cfg.encoder.patch_size**2 * cfg.encoder.in_chans
    )
    if arm in ("gmm_full", "gmm_full_private_decoder"):
        return float(
            (4 * (patch_projection + depth * dense) + (4 if arm.endswith("private_decoder") else 1) * decoder) / 1e9
        )
    if arm == "gmm_split":
        return float((patch_projection + (2 + 4 * (depth - 2)) * dense + decoder) / 1e9)
    moe_layers = len(cfg.encoder.moe_blocks) if cfg.encoder.moe_blocks is not None else min(depth, 2)
    active_mlps = 1 if arm == "vmoe_top1" else 4
    router = 2 * width * 4 if arm.startswith("vmoe") else 0
    return float(
        (
            patch_projection
            + (depth - moe_layers) * dense
            + moe_layers * (block(visible, width, hidden, active_mlps) + router)
            + decoder
        )
        / 1e9
    )


def _update(
    method: MAE,
    optimizer: torch.optim.Optimizer,
    views: torch.Tensor,
    noise: torch.Tensor,
    probabilities: torch.Tensor,
    cfg: DictConfig,
    seed: int,
    step: int,
    arm: str,
) -> dict[str, Any]:
    method.train()
    optimizer.zero_grad(set_to_none=True)
    observer = None if isinstance(method, TrunkMixtureMAE) else StepRoutingReadout(method)
    try:
        with (
            isolated_rng(keyed_seed("router_noise", seed, step), str(cfg.device)),
            _conditioning(method, probabilities),
        ):
            losses = method.loss_components(views.to(cfg.device), augmented=True, mask_noise=noise)
            losses.total.backward()
            if isinstance(method, TrunkMixtureMAE):
                posterior = method.probabilities
                if posterior is None:
                    raise RuntimeError("Missing trunk routing probabilities.")
                mass = posterior.mean(0)
                readout = {"l0_mean_probabilities": mass.tolist(), "l0_batch_top1_expert": int(mass.argmax())}
            else:
                readout = observer.rows if observer is not None else {}
        grad = torch.nn.utils.clip_grad_norm_(method.parameters(), float(cfg.grad_clip), error_if_nonfinite=True)
        if not bool(torch.isfinite(losses.total)):
            raise FloatingPointError("Non-finite MAE loss.")
        rates = apply_expert_learning_rates(optimizer, readout) if arm.startswith("gmm") else {}
        optimizer.step()
        return {
            "reconstruction_mse": float(losses.reconstruction.detach()),
            "auxiliary": float(losses.auxiliary.detach()),
            "gradient_l2": float(grad),
            **readout,
            **rates,
        }
    finally:
        if observer is not None:
            observer.close()


def run_arm(  # noqa: C901 - keep checkpoint and log commit ordering together
    cfg: DictConfig,
    data: ExperimentData,
    panel: PreparedPanel,
    seed: int,
    arm: str,
    directory: Path,
    protocol_hash: str,
    stop_after: int | None = None,
) -> dict[str, Any]:
    """Run a seed/arm with a resumable optimizer and an exact pre-night checkpoint."""
    directory.mkdir(parents=True, exist_ok=True)
    complete_path = directory / "complete.json"
    if complete_path.exists():
        result: dict[str, Any] = json.loads(complete_path.read_text())
        if result["protocol_hash"] != protocol_hash:
            raise ValueError("Completed run belongs to a different protocol.")
        return result

    method = build_arm(cfg, seed, arm)
    initial_hash = state_hash(method.state_dict())
    optimizer = instantiate(cfg.optim, _partial_=True)(params=optimizer_groups(method, arm, float(cfg.optim.lr)))
    orders = phase_orders(data, cfg, seed)
    batches = [(phase, ids) for phase, order in enumerate(orders) for ids in order.split(int(cfg.batch_size))]
    boundary = math.ceil(len(orders[0]) / int(cfg.batch_size))
    checkpoint = directory / "latest.pt"
    paths = {name: directory / f"{name}.jsonl" for name in ("training", "evaluation")}
    step, elapsed = 0, 0.0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=cfg.device, weights_only=False)
        if state["protocol_hash"] != protocol_hash or state["initial_hash"] != initial_hash:
            raise ValueError("Checkpoint belongs to a different protocol or initialization.")
        method.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, elapsed = state["step"], state["seconds"]
        for name, path in paths.items():
            with path.open("r+b") as stream:
                stream.truncate(state["offsets"][name])
    else:
        for path in paths.values():
            path.write_text("")
        _atomic_checkpoint(directory / "initial.pt", {"model": method.state_dict()})

    with paths["training"].open("a") as train, paths["evaluation"].open("a") as validation:

        def measure() -> None:
            metrics = evaluate(method, panel, str(cfg.device))
            validation.write(json.dumps({"step": step, **metrics}, allow_nan=False) + "\n")
            validation.flush()
            train.flush()
            if step == boundary:
                _atomic_checkpoint(directory / "pre_night.pt", {"model": method.state_dict(), "step": step})
            _atomic_checkpoint(
                checkpoint,
                {
                    "protocol_hash": protocol_hash,
                    "initial_hash": initial_hash,
                    "model": method.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "seconds": elapsed,
                    "offsets": {"training": train.tell(), "evaluation": validation.tell()},
                },
            )
            logger.info(
                "seed={} arm={} step={}/{} day+dusk={:.5f} night={:.5f}",
                seed,
                arm,
                step,
                len(batches),
                metrics["day_dusk_mse"],
                metrics["night_mse"],
            )

        if not checkpoint.exists():
            measure()
        for phase, ids in batches[step:]:
            started = time.perf_counter()
            views, noise = paired_views(method, data.corpus.images[ids], ids, seed, phase)
            signals = _update(method, optimizer, views, noise, data.probabilities[ids], cfg, seed, step, arm)
            synchronize(str(cfg.device))
            elapsed += time.perf_counter() - started
            step += 1
            row = {
                "step": step,
                "phase": PHASES[phase],
                "sample_ids": ids.tolist(),
                "views_hash": tensor_hash(views),
                "mask_noise_hash": tensor_hash(noise),
                **signals,
            }
            train.write(json.dumps(row, allow_nan=False) + "\n")
            if step % int(cfg.eval_every) == 0 or step in (boundary, boundary + 1, len(batches), stop_after):
                measure()
            if step == stop_after:
                return {"step": step}

    evaluations = [json.loads(line) for line in paths["evaluation"].read_text().splitlines()]
    pre_night = next(row for row in evaluations if row["step"] == boundary)
    result = {
        "protocol_hash": protocol_hash,
        "initial_hash": initial_hash,
        "arm": arm,
        "seed": seed,
        "steps": step,
        "boundary_step": boundary,
        "images_per_phase": dict(zip(PHASES, map(len, orders), strict=True)),
        "parameters": sum(parameter.numel() for parameter in method.parameters()),
        "approx_forward_gflops_per_image": forward_gflops(cfg, arm),
        "pre_night": pre_night,
        "final": evaluations[-1],
        "update_seconds": elapsed,
        "forgetting_mse": evaluations[-1]["day_dusk_mse"] - pre_night["day_dusk_mse"],
        "peak_forgetting_mse": max(row["day_dusk_mse"] for row in evaluations if row["step"] >= boundary)
        - pre_night["day_dusk_mse"],
    }
    write_json(complete_path, result)
    return result


@dataclass
class WorkerState:
    """Verified inputs loaded once per worker."""

    cfg: DictConfig
    data: ExperimentData
    panel: PreparedPanel


_worker: WorkerState | None = None


def _initialize_worker(config: dict[str, Any], output: Path) -> None:
    global _worker  # noqa: PLW0603
    cfg = OmegaConf.create(config)
    torch.set_num_threads(int(cfg.threads))
    protocol = json.loads((output / "protocol.json").read_text())
    for name, digest in protocol["source_hashes"].items():
        if digest_file(Path(__file__).parent / name) != digest:
            raise ValueError(f"Frozen training source changed: {name}")
    data = load_data(cfg)
    panel = prepare_panel(cfg, data)
    provenance = json.loads(json.dumps(data.provenance))
    if (
        data.corpus.fingerprint != protocol["data"]
        or provenance != protocol["conditioning"]
        or panel.fingerprint != protocol["validation_views_and_masks_sha256"]
    ):
        raise ValueError("Worker inputs do not match the frozen protocol.")
    _worker = WorkerState(cfg, data, panel)


def _run_job(output: Path, seed: int, arm: str, fingerprint: str) -> dict[str, Any]:
    if _worker is None:
        raise RuntimeError("Worker inputs were not initialized.")
    return run_arm(_worker.cfg, _worker.data, _worker.panel, seed, arm, output / f"seed_{seed}" / arm, fingerprint)


def run_experiment(cfg: DictConfig, output: Path) -> dict[str, Any]:
    """Freeze inputs, run six paired arms per seed, and audit their comparison."""
    if not cfg.seeds or len(set(cfg.seeds)) != len(cfg.seeds):
        raise ValueError("Use a nonempty list of distinct paired seeds.")
    if min(int(cfg.batch_size), int(cfg.eval_every), int(cfg.mask_draws), int(cfg.workers), int(cfg.threads)) < 1:
        raise ValueError("Batch size, evaluation cadence, draws, workers and threads must be positive.")
    if cfg.encoder.depth < 3 or cfg.encoder.num_experts != 4:
        raise ValueError("The split trunk requires two shared blocks and private suffixes with four experts.")
    torch.set_num_threads(int(cfg.threads))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = load_data(cfg)
    panel = prepare_panel(cfg, data)
    config = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    source_dir = Path(__file__).parent
    files = [
        Path(__file__),
        source_dir / "cluster_routing_experiment.py",
        source_dir / "expert_learning_rates.py",
        source_dir / "routing_experiment.py",
        source_dir / "routing_streams.py",
        *sorted((source_dir / "models").glob("*.py")),
        *sorted((source_dir / "ssl").glob("*.py")),
    ]
    protocol = {
        "experiment": cfg.experiment_id,
        "arms": list(ARMS),
        "config": config,
        "data": data.corpus.fingerprint,
        "conditioning": data.provenance,
        "torch": torch.__version__,
        "validation_views_and_masks_sha256": panel.fingerprint,
        "validation_counts": {
            "day": int((panel.labels == 0).sum()),
            "dusk": int((panel.labels == 1).sum()),
            "night": int((panel.labels == 2).sum()),
        },
        "phase_orders": {
            str(seed): {
                phase: {"images": len(order), "sha256": tensor_hash(order)}
                for phase, order in zip(PHASES, phase_orders(data, cfg, int(seed)), strict=True)
            }
            for seed in cfg.seeds
        },
        "source_hashes": {str(path.relative_to(source_dir)): digest_file(path) for path in files},
    }
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text())["protocol_hash"] != fingerprint:
        raise ValueError("Refusing to mix different frozen protocols. Use a fresh output directory.")
    write_json(protocol_path, {**protocol, "protocol_hash": fingerprint})
    for source in files:
        target = output / "provenance" / source.relative_to(source_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    jobs = [(int(seed), arm) for arm in reversed(ARMS) for seed in cfg.seeds]
    if int(cfg.workers) > 1:
        with ProcessPoolExecutor(
            max_workers=int(cfg.workers),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(config, output),
        ) as executor:
            futures = [executor.submit(_run_job, output, seed, arm, fingerprint) for seed, arm in jobs]
            for future in as_completed(futures):
                future.result()
    else:
        for seed, arm in jobs:
            run_arm(cfg, data, panel, seed, arm, output / f"seed_{seed}" / arm, fingerprint)
    from cafl4ds.day_night_analysis import analyze  # noqa: PLC0415

    return analyze(output, data.probabilities)
