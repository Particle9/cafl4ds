"""P1.4.0 fixed-split, fixed-budget local-CPU experiment orchestration."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
import shutil
import subprocess
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from cafl4ds import harness
from cafl4ds.data.sources import DataSource, STL10Source, SyntheticSource
from cafl4ds.data.streams import EraStream, EvalSets, SplitIndices
from cafl4ds.filters.random import RandomCount
from cafl4ds.filters.reservoir import FixedBudgetReservoir
from cafl4ds.filters.study import StudyAcceptAll, StudyLossHalf
from cafl4ds.jsonio import dumps_valid
from cafl4ds.monitor import HealthMonitor
from cafl4ds.ssl.base import SSLMethod


class TensorSource(DataSource):
    """Reuse one source load across independent arms without copying its images."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        """Hold resized image tensors and source labels without another copy."""
        self.images, self.labels = images, labels

    @property
    def num_classes(self) -> int:
        """Number of labelled source classes."""
        return int(torch.unique(self.labels).numel())

    def load(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the same source tensors for independent stream construction."""
        return self.images, self.labels


def _sha_json(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _source_fingerprint(config: DictConfig) -> dict[str, object]:
    if config.data_kind == "synthetic":
        return {
            "kind": "synthetic",
            "seed": int(config.seed),
            "per_class": int(config.smoke.per_class),
            "classes": int(config.smoke.classes),
        }
    folder = Path(config.data_root) / "stl10_binary"
    return {
        "kind": "stl10_train",
        "files": {
            name: {"size": (folder / name).stat().st_size, "sha256": _sha_file(folder / name)}
            for name in ("train_X.bin", "train_y.bin")
        },
    }


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    """Hash every model tensor, including BatchNorm buffers, in stable key order."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _code_fingerprint() -> dict[str, object]:
    root = Path(__file__).resolve().parent.parent
    files = sorted((root / "cafl4ds").rglob("*.py")) + sorted((root / "scripts").glob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    git = shutil.which("git")
    revision = (
        subprocess.run(  # noqa: S603 - executable resolved from PATH; fixed read-only arguments
            [git, "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
        ).stdout.strip()
        if git
        else ""
    )
    dirty_diff = (
        subprocess.run(  # noqa: S603 - fixed read-only Git inspection
            [git, "diff", "--binary", "--", "cafl4ds", "scripts", "pyproject.toml"],
            cwd=root,
            capture_output=True,
            check=False,
        ).stdout
        if git
        else b""
    )
    return {
        "revision": revision,
        "python_code_sha256": digest.hexdigest(),
        "dirty_diff_sha256": hashlib.sha256(dirty_diff).hexdigest(),
    }


def _effective(config: DictConfig) -> dict[str, int | str]:
    smoke = config.profile == "smoke"
    if smoke and config.data_kind != "synthetic":
        raise ValueError("smoke profile requires data_kind=synthetic")
    if config.profile not in {"matched_384", "legacy_400", "smoke"}:
        raise ValueError(f"unknown profile {config.profile}")
    if not smoke and config.data_kind != "stl10":
        raise ValueError("real profiles require data_kind=stl10")
    small = config.smoke
    values: dict[str, int | str] = {}
    for key in (
        "img_size",
        "batch_size",
        "train_per_class",
        "support_per_class",
        "query_per_class",
        "era_eval_per_class",
        "epochs",
        "quota_half",
    ):
        values[key] = int(small[key] if smoke else config[key])
    if config.profile == "legacy_400":
        values["train_per_class"] = 400
    if not smoke and (values["batch_size"] != 128 or values["train_per_class"] not in {384, 400}):
        raise ValueError("real P1.4.0 profiles require the registered batch and class counts")
    if int(values["quota_half"]) * 2 != int(values["batch_size"]):
        raise ValueError("half-budget policy must use exactly half the incoming batch")
    return values


def load_source(config: DictConfig, values: dict[str, int | str]) -> TensorSource:
    """Load the real or synthetic source once per invocation."""
    if config.data_kind == "stl10":
        source: DataSource = STL10Source(root=str(config.data_root), split="train", img_size=int(values["img_size"]))
    else:
        source = SyntheticSource(
            num_classes=int(config.smoke.classes),
            per_class=int(config.smoke.per_class),
            img_size=int(values["img_size"]),
            seed=int(config.seed),
        )
    return TensorSource(*source.load())


def seed_components(seed: int) -> dict[str, int]:
    """Separate split, order, model, training, selection and evaluation RNGs."""
    return {
        "split_seed": 100_000 + seed,
        "order_seed": 200_000 + seed,
        "init_seed": 300_000 + seed,
        "train_seed": 400_000 + seed,
        "selection_seed": 500_000 + seed,
        "eval_seed": 600_000 + seed,
    }


def _splits(source: TensorSource, values: dict[str, int | str], seeds: dict[str, int]) -> SplitIndices:
    return SplitIndices.create(
        source.labels,
        seed=seeds["split_seed"],
        support_per_class=int(values["support_per_class"]),
        query_per_class=int(values["query_per_class"]),
        era_eval_per_class=int(values["era_eval_per_class"]),
        max_train_per_class=int(values["train_per_class"]),
    )


def make_stream(
    source: TensorSource,
    values: dict[str, int | str],
    seeds: dict[str, int],
    splits: SplitIndices | None,
    ordering: str,
) -> EraStream:
    """Create a stream with shared partitions and a reproducible class permutation."""
    classes = sorted(set(source.labels.tolist()))
    if ordering == "iid":
        order, block, class_order = "iid", None, classes
    elif ordering in {"b128", "b256", "full"}:
        order = "class_blocked"
        block = {"b128": 128, "b256": 256, "full": None}[ordering]
        if values["batch_size"] != 128 and ordering != "full":
            raise ValueError("b128/b256 are registered only for batch size 128")
        perm = torch.randperm(len(classes), generator=torch.Generator().manual_seed(seeds["order_seed"]))
        class_order = classes if splits is None else [classes[i] for i in perm.tolist()]
    else:
        raise ValueError(f"unknown ordering {ordering}")
    stream = EraStream(
        source=source,
        batch_size=int(values["batch_size"]),
        order=order,
        class_order=class_order,
        block_size=block,
        support_per_class=int(values["support_per_class"]),
        query_per_class=int(values["query_per_class"]),
        era_eval_per_class=int(values["era_eval_per_class"]),
        max_train_per_class=int(values["train_per_class"]),
        seed=seeds["order_seed"],
        split_indices=splits,
    )
    if splits is not None:
        reference_ids = set().union(*(set(ids) for ids in splits.train.values()))
        if set(stream.ordered_ids) != reference_ids or len(stream.ordered_ids) != len(reference_ids):
            raise AssertionError("ordering changed the fixed training population")
    if int(values["train_per_class"]) % int(values["batch_size"]) == 0:
        if any(len(ids) != int(values["batch_size"]) for ids in stream.batch_ids):
            raise AssertionError("registered matched profile produced a partial incoming batch")
        expected = len(stream.ordered_ids) // int(values["batch_size"])
        if len(stream) != expected:
            raise AssertionError("ordering changed the optimizer-update count")
    return stream


class StudyMonitor(HealthMonitor):
    """Collapse geometry each checkpoint and a common ten-way probe on selected epochs."""

    def __init__(
        self,
        *,
        eval_sets: EvalSets,
        out_dir: Path,
        steps_per_epoch: int,
        linear_epochs: set[int],
        knn_k: int = 20,
        run_knn: bool = True,
        run_alignment: bool = True,
        align_seed: int = 0,
    ) -> None:
        """Pin probe settings and checkpoint schedule for this study."""
        super().__init__(
            eval_sets=eval_sets,
            knn_k=knn_k,
            run_knn=run_knn,
            run_alignment=run_alignment,
            align_seed=align_seed,
            run_linear=False,
        )
        self.out_dir = out_dir
        self.steps_per_epoch = steps_per_epoch
        self.linear_epochs = linear_epochs

    def measure(self, method: SSLMethod, step: int) -> dict[str, float]:
        """Read health and cache support/query features at declared probe epochs."""
        metrics = super().measure(method, step)
        epoch = 0 if step < 0 else (step + 1) // self.steps_per_epoch
        if epoch not in self.linear_epochs:
            return metrics
        was_training = method.training
        method.eval()
        try:
            with torch.no_grad():
                support = method.encode(self.eval_sets.probe_support.images).cpu().numpy()
                query = method.encode(self.eval_sets.probe_query.images).cpu().numpy()
        finally:
            method.train(was_training)
        support_labels = self.eval_sets.probe_support.labels.numpy()
        query_labels = self.eval_sets.probe_query.labels.numpy()
        scaler = StandardScaler().fit(support)
        with warnings.catch_warnings(record=True) as convergence:
            warnings.simplefilter("always")
            classifier = LogisticRegression(C=1.0, max_iter=1000).fit(scaler.transform(support), support_labels)
        predictions = classifier.predict(scaler.transform(query))
        metrics["linear_acc"] = float(np.mean(predictions == query_labels))
        metrics["linear_convergence_warnings"] = float(len(convergence))
        for label in sorted(set(query_labels.tolist())):
            rows = query_labels == label
            metrics[f"class_{label}_acc"] = float(np.mean(predictions[rows] == label))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.out_dir / f"features_epoch_{epoch:03d}.npz",
            support=support,
            query=query,
            support_labels=support_labels,
            query_labels=query_labels,
            predictions=predictions,
        )
        return metrics


def _selector(
    policy: str,
    values: dict[str, int | str],
    seeds: dict[str, int],
    stream: EraStream,
    *,
    reservoir_capacity: int,
) -> StudyAcceptAll | RandomCount | StudyLossHalf | FixedBudgetReservoir:
    if policy == "accept_all":
        return StudyAcceptAll()
    if policy == "random_half":
        return RandomCount(count=int(values["quota_half"]), seed=seeds["selection_seed"])
    if policy == "loss_half":
        return StudyLossHalf(count=int(values["quota_half"]), seed=seeds["selection_seed"])
    if policy == "replay_fixed":
        return FixedBudgetReservoir(
            incoming_count=int(values["batch_size"]),
            replay_count=int(values["quota_half"]),
            capacity=reservoir_capacity,
            seed=seeds["selection_seed"],
            arrival_batches=stream.batch_ids,
        )
    raise ValueError(f"unknown policy {policy}")


def _arm_provenance(
    config: DictConfig,
    values: dict[str, int | str],
    source_fp: dict[str, object],
    code_fp: dict[str, object],
    splits: SplitIndices | None,
    stream: EraStream,
    *,
    seed: int,
    ordering: str,
    policy: str,
    pc: bool,
) -> dict[str, object]:
    protocol = {
        "profile": config.profile,
        "values": values,
        "seed": seed,
        "ordering": ordering,
        "policy": policy,
        "pc": pc,
        "seeds": seed_components(seed),
        "model": OmegaConf.to_container(config.ssl, resolve=True),
        "encoder": OmegaConf.to_container(config.encoder, resolve=True),
        "optim": OmegaConf.to_container(config.optim, resolve=True),
        "schedule": OmegaConf.to_container(config.schedule, resolve=True),
        "linear_epochs": list(config.eval_linear_epochs),
        "source": source_fp,
        "code": code_fp,
        "split_hash": _sha_json(asdict(splits)) if splits is not None else None,
        "ordered_ids_hash": _sha_json(stream.ordered_ids),
        "batch_ids_hash": _sha_json(stream.batch_ids),
        "device": config.device,
        "threads": int(config.num_threads),
        "reservoir_capacity": 16 if config.profile == "smoke" else int(config.reservoir_capacity),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "machine": platform.machine(),
        },
        "study_protocol_hash": config.get("study_protocol_hash"),
    }
    protocol["hash"] = _sha_json(protocol)
    return protocol


def run_arm(
    config: DictConfig,
    source: TensorSource,
    values: dict[str, int | str],
    source_fp: dict[str, object],
    code_fp: dict[str, object],
    *,
    seed: int,
    ordering: str,
    policy: str,
    pc: bool = False,
    output_root: Path,
) -> dict[str, Any]:
    """Run or safely reuse one completely specified experimental arm."""
    seeds = seed_components(seed)
    splits = None if config.profile == "legacy_400" else _splits(source, values, seeds)
    stream = make_stream(source, values, seeds, splits, ordering)
    provenance = _arm_provenance(
        config, values, source_fp, code_fp, splits, stream, seed=seed, ordering=ordering, policy=policy, pc=pc
    )
    name = f"{config.profile}_seed{seed}_{ordering}_{policy}{'_pc' if pc else ''}"
    out_dir = output_root / name
    comparison_path = out_dir / "comparison.json"
    if comparison_path.exists():
        prior = json.loads(comparison_path.read_text())
        if prior.get("collapse_diet", {}).get("provenance", {}).get("hash") == provenance["hash"]:
            if not prior["gate"]["passed"]:
                raise ValueError(f"previous arm is scientifically invalid: {comparison_path}")
            return cast("dict[str, Any]", prior)
        raise ValueError(f"existing arm has different provenance: {comparison_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(
        dumps_valid(
            {
                "provenance": provenance,
                "split": asdict(splits) if splits is not None else None,
                "class_order": stream.class_order,
                "ordered_ids": stream.ordered_ids,
                "batch_ids": stream.batch_ids,
            }
        ),
        encoding="utf-8",
    )
    status_path = out_dir / "status.json"
    status_path.write_text(
        dumps_valid(
            {"state": "incomplete", "provenance_hash": provenance["hash"], "resume_mode": "restart_from_initial"}
        ),
        encoding="utf-8",
    )
    torch.manual_seed(seeds["init_seed"])
    method = instantiate(
        config.ssl, encoder=instantiate(config.encoder, img_size=values["img_size"]), anti_collapse=not pc
    )
    b5_method = copy.deepcopy(method) if not pc else None
    initial_state = copy.deepcopy(method.state_dict())
    initial_hash = _state_hash(initial_state)
    torch.save(initial_state, out_dir / "initial.pt")
    torch.manual_seed(seeds["train_seed"])
    optimizer = instantiate(config.optim, params=method.parameters())
    total_steps = int(values["epochs"]) * len(stream)
    scheduler = instantiate(config.schedule, optimizer=optimizer, total_steps=total_steps)
    linear_epochs = set(config.eval_linear_epochs) | {0, int(values["epochs"])}
    monitor = StudyMonitor(
        eval_sets=stream.eval_sets,
        out_dir=out_dir,
        steps_per_epoch=len(stream),
        linear_epochs=linear_epochs,
        knn_k=20,
        run_knn=True,
        run_alignment=True,
        align_seed=seeds["eval_seed"],
    )
    selector = _selector(
        policy,
        values,
        seeds,
        stream,
        reservoir_capacity=16 if config.profile == "smoke" else int(config.reservoir_capacity),
    )
    arm = harness.run_stream_arm(
        name=name,
        role="pc" if pc else "live",
        method=method,
        stream=stream,
        optimizer=optimizer,
        selection_filter=selector,
        monitor=monitor,
        out_dir=out_dir,
        eval_every=len(stream),
        epochs=int(values["epochs"]),
        scheduler=scheduler,
        device="cpu",
        measure_initial=True,
        epoch_end_only=True,
        log_selection=True,
    )
    torch.save(
        {"model": method.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        out_dir / "final.pt",
    )
    b5 = None
    if b5_method is not None:
        b5_monitor = StudyMonitor(
            eval_sets=stream.eval_sets,
            out_dir=out_dir / "b5",
            steps_per_epoch=len(stream),
            linear_epochs={0},
            knn_k=20,
            run_knn=True,
            run_alignment=True,
            align_seed=seeds["eval_seed"],
        )
        b5 = harness.run_frozen_arm(
            name=f"{name}_b5", frozen_method=b5_method, monitor=b5_monitor, grid=harness.health_grid(arm), device="cpu"
        )
    records = arm.records
    decisions = [record for record in records if record.get("series") == "selection"]
    budget_ok = (
        len(decisions) == total_steps
        and len(arm.loss_records) == total_steps
        and all(
            row["raw_count"] == len(stream.batch_ids[int(row["step"]) % len(stream)])
            and row["trained"]
            == (int(values["quota_half"]) if policy in {"random_half", "loss_half"} else row["raw_count"])
            and len(row["current_rows"]) + len(row["replay_events"]) == row["trained"]
            and not row["skipped"]
            for row in decisions
        )
    )
    budget_ok = budget_ok and all(
        int(row["trained"]) == int(decision["trained"]) and row["optimizer_step"] == index + 1
        for index, (row, decision) in enumerate(zip(arm.loss_records, decisions, strict=True))
    )
    full_trace = []
    for row in decisions:
        batch_ids = stream.batch_ids[int(row["step"]) % len(stream)]
        full_trace.append({**row, "current_source_ids": [batch_ids[i] for i in row["current_rows"]]})
    (out_dir / "selection.json").write_text(dumps_valid(full_trace), encoding="utf-8")
    finite = harness.all_finite(arm, *((b5,) if b5 is not None else ()))
    gate = {
        "mode": "apparatus",
        "checks": {
            "budget": budget_ok,
            "finite": finite,
            "expected_checkpoints": len(arm.health) == int(values["epochs"]) + 1,
        },
        "passed": bool(budget_ok and finite and len(arm.health) == int(values["epochs"]) + 1),
    }
    comparison = harness.build_comparison(
        config_header={
            "C": "simsiam_collapse" if pc else "simsiam",
            "I": "from_scratch",
            "F": ordering,
            "A": policy,
            "seed": seed,
            "alignment": "fixed-split-sequential",
        },
        gate=gate,
        arms=[arm] + ([b5] if b5 is not None else []),
        extensions={
            "collapse_diet": {
                "provenance": provenance,
                "profile": config.profile,
                "budget": {
                    "raw_arrivals": sum(int(row["raw_count"]) for row in decisions),
                    "trained_presentations": sum(int(row["trained"]) for row in decisions),
                    "updates": len(arm.loss_records),
                    "skipped_updates": sum(bool(row["skipped"]) for row in decisions),
                    "selection_seconds": sum(float(row["selection_seconds"]) for row in decisions),
                },
                "pc": pc,
                "loss_floor": arm.loss_floor,
                "initial_state_hash": initial_hash,
                "references": {
                    "matched_iid": f"{config.profile}_seed{seed}_iid_{policy}",
                    "frozen_initial": f"{name}_b5" if b5 is not None else None,
                    "frozen_final_iid": f"{config.profile}_seed{seed}_iid_{policy}/final.pt",
                    "iid_positive_control": f"{config.profile}_seed{seed}_iid_{policy}_pc",
                },
                "status": "complete" if gate["passed"] else "invalid",
            }
        },
    )
    comparison_path.write_text(dumps_valid(comparison), encoding="utf-8")
    status_path.write_text(
        dumps_valid({"state": "complete" if gate["passed"] else "invalid", "provenance_hash": provenance["hash"]}),
        encoding="utf-8",
    )
    if not gate["passed"]:
        raise ValueError(f"apparatus invalid in {comparison_path}: {gate['checks']}")
    return comparison


def late_metric(comparison: dict[str, Any], key: str, role: str = "live") -> float | None:
    """Mean of the last five measured epochs, excluding the initial read."""
    rows = [row[key] for row in comparison["arms"][role]["health"] if row.get("epoch", 0) > 0 and key in row]
    return float(np.mean(rows[-5:])) if rows else None


def _metric(comparison: dict[str, Any], key: str, role: str = "live") -> float:
    value = late_metric(comparison, key, role)
    if value is None:
        raise ValueError(f"missing {key} in {role} of {comparison['config']}")
    return value


def bootstrap_interval(values: list[float], *, draws: int = 10000, seed: int = 42) -> tuple[float, float]:
    """Seed-level percentile interval; checkpoint and query rows are not replicates."""
    if not values:
        raise ValueError("cannot bootstrap an empty effect")
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    means = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def sign_flip_pvalue(values: list[float]) -> float:
    """Exact two-sided paired sign-flip test for at most twenty seed pairs."""
    if not 1 <= len(values) <= 20:
        raise ValueError("sign-flip test expects one to twenty seed effects")
    observed = abs(sum(values))
    count = sum(
        abs(sum(sign * value for sign, value in zip(signs, values, strict=True))) >= observed - 1e-12
        for signs in itertools.product((-1, 1), repeat=len(values))
    )
    return float(count / (2 ** len(values)))


def paired_effect(values: list[float], *, draws: int = 10000) -> dict[str, Any]:
    """Expose all seed effects and uncertainty; never collapse a null into a pass."""
    lo, hi = bootstrap_interval(values, draws=draws)
    return {
        "by_seed": values,
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": min(values),
        "max": max(values),
        "ci95": [lo, hi],
        "same_positive_sign": sum(v > 0 for v in values),
        "sign_flip_p": sign_flip_pvalue(values),
    }


def _pair(
    results: dict[tuple[int, str, str, bool], dict[str, Any]],
    seeds: list[int],
    left: tuple[str, str],
    right: tuple[str, str],
    key: str,
    *,
    log_ratio: bool = False,
    draws: int = 10000,
) -> dict[str, Any]:
    effects = []
    for seed in seeds:
        lhs = _metric(results[(seed, *left, False)], key)
        rhs = _metric(results[(seed, *right, False)], key)
        effects.append(math.log(lhs / rhs) if log_ratio else lhs - rhs)
    return paired_effect(effects, draws=draws)


def _pair_final(
    results: dict[tuple[int, str, str, bool], dict[str, Any]],
    seeds: list[int],
    left: tuple[str, str],
    right: tuple[str, str],
    key: str,
    *,
    log_ratio: bool = False,
    draws: int = 10000,
) -> dict[str, Any]:
    """Secondary final-checkpoint paired effect, separate from the late mean."""
    effects = []
    for seed in seeds:
        lhs = float(results[(seed, *left, False)]["arms"]["live"]["health"][-1][key])
        rhs = float(results[(seed, *right, False)]["arms"]["live"]["health"][-1][key])
        effects.append(math.log(lhs / rhs) if log_ratio else lhs - rhs)
    return paired_effect(effects, draws=draws)


def _reference_checks(
    results: dict[tuple[int, str, str, bool], dict[str, Any]], seeds: list[int], policy: str, draws: int
) -> dict[str, Any]:
    rows = []
    improvements = []
    for seed in seeds:
        healthy = results[(seed, "iid", policy, False)]
        pc = results[(seed, "iid", policy, True)]
        loss_floor = pc["collapse_diet"].get("loss_floor")
        rank_ratio = _metric(healthy, "rankme_proj") / _metric(pc, "rankme_proj", role="pc")
        quality_gain = _metric(healthy, "linear_acc") - float(healthy["arms"]["b5"]["health"][0]["linear_acc"])
        init_match = healthy["collapse_diet"]["initial_state_hash"] == pc["collapse_diet"]["initial_state_hash"]
        rows.append(
            {
                "seed": seed,
                "pc_loss_floor": loss_floor,
                "rank_ratio": rank_ratio,
                "quality_gain_vs_b5": quality_gain,
                "initial_state_match": init_match,
                "pc_pass": loss_floor is not None and loss_floor <= -0.9 and rank_ratio >= 2.0 and init_match,
            }
        )
        improvements.append(quality_gain)
    gain = paired_effect(improvements, draws=draws)
    return {
        "rows": rows,
        "learning": gain,
        "all_pc_pass": all(bool(row["pc_pass"]) for row in rows),
        "development_learning_pass": float(gain["mean"]) >= 0.03
        and sum(v > 0 for v in improvements) >= max(1, len(improvements) - 1),
        "confirmation_learning_pass": gain["ci95"][0] > 0,
    }


def analyze_susceptibility(
    results: dict[tuple[int, str, str, bool], dict[str, Any]],
    seeds: list[int],
    *,
    margin: float,
    geometry_ratio: float,
    draws: int,
) -> dict[str, Any]:
    """Choose a stress condition on unfiltered development data only."""
    reference = _reference_checks(results, seeds, "accept_all", draws)
    cells: dict[str, object] = {}
    chosen = None
    for ordering in ("b128", "b256", "full"):
        initial_match = all(
            results[(seed, ordering, "accept_all", False)]["collapse_diet"]["initial_state_hash"]
            == results[(seed, "iid", "accept_all", False)]["collapse_diet"]["initial_state_hash"]
            for seed in seeds
        )
        quality = _pair(results, seeds, ("iid", "accept_all"), (ordering, "accept_all"), "linear_acc", draws=draws)
        geometry = _pair(
            results, seeds, ("iid", "accept_all"), (ordering, "accept_all"), "rankme_proj", log_ratio=True, draws=draws
        )
        qualifies = (
            initial_match
            and quality["mean"] >= margin
            and geometry["mean"] >= math.log(geometry_ratio)
            and quality["same_positive_sign"] >= 4
            and geometry["same_positive_sign"] >= 4
        )
        cells[ordering] = {
            "quality": quality,
            "geometry": geometry,
            "final_quality": _pair_final(
                results, seeds, ("iid", "accept_all"), (ordering, "accept_all"), "linear_acc", draws=draws
            ),
            "final_geometry": _pair_final(
                results,
                seeds,
                ("iid", "accept_all"),
                (ordering, "accept_all"),
                "rankme_proj",
                log_ratio=True,
                draws=draws,
            ),
            "initial_state_match": initial_match,
            "qualifies": qualifies,
        }
        if chosen is None and qualifies:
            chosen = ordering
    return {
        "reference": reference,
        "cells": cells,
        "selected_stress": chosen if reference["all_pc_pass"] and reference["development_learning_pass"] else None,
    }


def analyze_selection(
    results: dict[tuple[int, str, str, bool], dict[str, Any]],
    seeds: list[int],
    stress: str,
    *,
    draws: int,
    quality_margin: float,
    geometry_ratio: float,
    confirmation: bool,
) -> dict[str, Any]:
    """Budget-matched admission and replay effects with IID no-harm checks."""
    refs = {policy: _reference_checks(results, seeds, policy, draws) for policy in ("accept_all", "random_half")}
    families = {
        "admission": ("loss_half", "random_half"),
        "replay": ("replay_fixed", "accept_all"),
    }
    effects: dict[str, dict[str, Any]] = {}
    for family, (policy, control) in families.items():
        quality = _pair(results, seeds, (stress, policy), (stress, control), "linear_acc", draws=draws)
        geometry = _pair(
            results, seeds, (stress, policy), (stress, control), "rankme_proj", log_ratio=True, draws=draws
        )
        knn = _pair(results, seeds, (stress, policy), (stress, control), "knn_acc", draws=draws)
        iid = _pair(results, seeds, ("iid", policy), ("iid", control), "linear_acc", draws=draws)
        specificity = [a - b for a, b in zip(quality["by_seed"], iid["by_seed"], strict=True)]
        residual_damage = _pair(results, seeds, ("iid", control), (stress, policy), "linear_acc", draws=draws)
        effects[family] = {
            "quality": quality,
            "geometry": geometry,
            "final_quality": _pair_final(
                results, seeds, (stress, policy), (stress, control), "linear_acc", draws=draws
            ),
            "final_geometry": _pair_final(
                results, seeds, (stress, policy), (stress, control), "rankme_proj", log_ratio=True, draws=draws
            ),
            "knn": knn,
            "iid_quality": iid,
            "stress_specificity": paired_effect(specificity, draws=draws),
            "residual_damage": residual_damage,
        }
    if confirmation:
        pvalues = [(family, float(effects[family]["quality"]["sign_flip_p"])) for family in families]
        sorted_ps = sorted(pvalues, key=lambda item: item[1])
        holm = {name: min(1.0, p * (2 - index)) for index, (name, p) in enumerate(sorted_ps)}
        holm[sorted_ps[1][0]] = max(holm[sorted_ps[1][0]], holm[sorted_ps[0][0]])
        for family in families:
            item = effects[family]
            q, r, iid = item["quality"], item["geometry"], item["iid_quality"]
            item["holm_p"] = holm[family]
            item["protection"] = bool(
                q["mean"] >= quality_margin
                and q["ci95"][0] > 0
                and holm[family] < 0.05
                and r["mean"] > 0
                and iid["ci95"][0] > -quality_margin
                and refs["random_half" if family == "admission" else "accept_all"]["confirmation_learning_pass"]
            )
            item["complete_rescue"] = bool(item["protection"] and item["residual_damage"]["mean"] <= quality_margin)
    return {"references": refs, "effects": effects}


def analyze_confirmation_diet(
    results: dict[tuple[int, str, str, bool], dict[str, Any]],
    seeds: list[int],
    stress: str,
    *,
    draws: int,
    quality_margin: float,
    geometry_ratio: float,
) -> dict[str, Any]:
    """Apply the locked susceptibility margins on held-out confirmation seeds."""
    quality = _pair(results, seeds, ("iid", "accept_all"), (stress, "accept_all"), "linear_acc", draws=draws)
    geometry = _pair(
        results, seeds, ("iid", "accept_all"), (stress, "accept_all"), "rankme_proj", log_ratio=True, draws=draws
    )
    quality_confirmed = quality["ci95"][0] > quality_margin
    geometry_confirmed = geometry["ci95"][0] > math.log(geometry_ratio)
    if quality_confirmed and geometry_confirmed:
        classification = "geometry_associated_degradation"
    elif quality_confirmed:
        classification = "quality_degradation_without_confirmed_geometry"
    elif geometry_confirmed:
        classification = "geometry_change_without_confirmed_quality_harm"
    else:
        classification = "not_confirmed"
    return {
        "quality": quality,
        "geometry": geometry,
        "final_quality": _pair_final(
            results, seeds, ("iid", "accept_all"), (stress, "accept_all"), "linear_acc", draws=draws
        ),
        "final_geometry": _pair_final(
            results, seeds, ("iid", "accept_all"), (stress, "accept_all"), "rankme_proj", log_ratio=True, draws=draws
        ),
        "classification": classification,
        "quality_confirmed": quality_confirmed,
        "geometry_confirmed": geometry_confirmed,
    }


def _load_existing(
    output_root: Path, seed: int, ordering: str, policy: str, *, profile: str, pc: bool = False
) -> dict[str, Any]:
    name = f"{profile}_seed{seed}_{ordering}_{policy}{'_pc' if pc else ''}"
    path = output_root / name / "comparison.json"
    if not path.exists():
        raise FileNotFoundError(f"required prior arm missing: {path}")
    result = json.loads(path.read_text())
    if not result["gate"]["passed"]:
        raise ValueError(f"required prior arm invalid: {path}")
    return cast("dict[str, Any]", result)


def _protocol(
    config: DictConfig,
    summary: dict[str, Any],
    output_root: Path,
    values: dict[str, int | str],
    source_fp: dict[str, object],
) -> dict[str, object]:
    protocol = {
        "stress": summary["selected_stress"],
        "quality_margin": float(config.quality_margin),
        "geometry_ratio": float(config.geometry_ratio),
        "seed_confirmation": list(config.seed_confirmation),
        "policy": ["accept_all", "random_half", "loss_half", "replay_fixed"],
        "scoring": "eval-mode-isolated-cpu",
        "quota_half": int(config.quota_half),
        "reservoir_capacity": int(config.reservoir_capacity),
        "analysis": "paired-bootstrap-10000-and-exact-sign-flip-holm",
        "output_root": str(output_root),
        "values": values,
        "source": source_fp,
        "encoder": OmegaConf.to_container(config.encoder, resolve=True),
        "ssl": OmegaConf.to_container(config.ssl, resolve=True),
        "optim": OmegaConf.to_container(config.optim, resolve=True),
        "schedule": OmegaConf.to_container(config.schedule, resolve=True),
        "eval_linear_epochs": list(config.eval_linear_epochs),
        "bootstrap_draws": int(config.bootstrap_draws),
    }
    protocol["hash"] = _sha_json(protocol)
    return protocol


def run_stage(config: DictConfig) -> dict[str, object]:  # noqa: C901 - explicit registered stage sequence
    """Run one registered stage, reusing only exact-provenance completed arms."""
    if config.device != "cpu":
        raise ValueError("P1.4.0 is currently certified for local CPU execution")
    torch.set_num_threads(int(config.num_threads))
    values = _effective(config)
    output_root = Path(config.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source = load_source(config, values)
    source_fp = _source_fingerprint(config)
    code_fp = _code_fingerprint()
    results: dict[tuple[int, str, str, bool], dict[str, Any]] = {}

    def collect(seed: int, ordering: str, policy: str, pc: bool = False) -> None:
        results[(seed, ordering, policy, pc)] = run_arm(
            config,
            source,
            values,
            source_fp,
            code_fp,
            seed=seed,
            ordering=ordering,
            policy=policy,
            pc=pc,
            output_root=output_root,
        )

    stage = str(config.stage)
    if stage == "smoke":
        if config.profile != "smoke":
            raise ValueError("smoke stage requires profile=smoke")
        for ordering, policy, pc in (
            ("iid", "accept_all", False),
            ("full", "accept_all", False),
            ("iid", "random_half", False),
            ("iid", "loss_half", False),
            ("full", "replay_fixed", False),
            ("iid", "accept_all", True),
        ):
            collect(int(config.seed), ordering, policy, pc)
        summary: dict[str, object] = {"stage": stage, "runs": len(results), "scientific_verdict": "not-applicable"}
    elif stage == "arm":
        collect(int(config.seed), str(config.ordering), str(config.policy), False)
        summary = {"stage": stage, "runs": len(results), "scientific_verdict": "not-evaluated"}
    elif stage == "bridge":
        if config.profile != "legacy_400":
            raise ValueError("bridge stage requires profile=legacy_400")
        for seed in (0, 1):
            for ordering, pc in (("iid", False), ("full", False), ("iid", True)):
                collect(seed, ordering, "accept_all", pc)
        summary = {"stage": stage, "runs": len(results), "scientific_verdict": "historical-bridge-only"}
    elif stage == "susceptibility":
        if config.profile != "matched_384":
            raise ValueError("susceptibility requires matched_384 profile")
        seeds = [int(seed) for seed in config.seed_development]
        for seed in seeds:
            for ordering in ("iid", "b128", "b256", "full"):
                collect(seed, ordering, "accept_all")
            collect(seed, "iid", "accept_all", True)
        summary = {
            "stage": stage,
            **analyze_susceptibility(
                results,
                seeds,
                margin=float(config.quality_margin),
                geometry_ratio=float(config.geometry_ratio),
                draws=int(config.bootstrap_draws),
            ),
        }
    elif stage in {"selection", "confirmation"}:
        if config.profile != "matched_384":
            raise ValueError("selection/confirmation require matched_384 profile")
        development_path = output_root / "susceptibility_summary.json"
        if not development_path.exists():
            raise FileNotFoundError(f"selection requires prior susceptibility results: {development_path}")
        development = json.loads(development_path.read_text())
        stress = development.get("selected_stress")
        if stress not in {"b128", "b256", "full"}:
            raise ValueError("no qualified stress condition; selection stage is not indicated")
        if stage == "confirmation":
            protocol_path = (
                Path(config.protocol_path).resolve() if config.protocol_path else output_root / "protocol.json"
            )
            protocol = json.loads(protocol_path.read_text())
            expected = _protocol(config, development, output_root, values, source_fp)
            if protocol != expected:
                raise ValueError("confirmation protocol disagrees with registered development choice")
            OmegaConf.update(config, "study_protocol_hash", protocol["hash"], force_add=True)
            seeds = [int(seed) for seed in protocol["seed_confirmation"]]
        else:
            seeds = [int(seed) for seed in config.seed_development]
        for seed in seeds:
            if stage == "selection":
                for ordering in ("iid", stress):
                    _load_existing(output_root, seed, ordering, "accept_all", profile=str(config.profile))
                    collect(seed, ordering, "accept_all")
            for ordering in ("iid", stress):
                for policy in (
                    ("random_half", "loss_half", "replay_fixed")
                    if stage == "selection"
                    else ("accept_all", "random_half", "loss_half", "replay_fixed")
                ):
                    collect(seed, ordering, policy)
            for policy in ("random_half",) if stage == "selection" else ("accept_all", "random_half"):
                collect(seed, "iid", policy, True)
            if stage == "selection":
                _load_existing(output_root, seed, "iid", "accept_all", profile=str(config.profile), pc=True)
                collect(seed, "iid", "accept_all", True)
        study = analyze_selection(
            results,
            seeds,
            stress,
            draws=int(config.bootstrap_draws),
            quality_margin=float(config.quality_margin),
            geometry_ratio=float(config.geometry_ratio),
            confirmation=stage == "confirmation",
        )
        summary = {"stage": stage, "stress": stress, **study}
        if stage == "confirmation":
            diet = analyze_confirmation_diet(
                results,
                seeds,
                stress,
                draws=int(config.bootstrap_draws),
                quality_margin=float(config.quality_margin),
                geometry_ratio=float(config.geometry_ratio),
            )
            summary["diet"] = diet
            summary["apparatus_valid"] = all(
                bool(ref["all_pc_pass"]) and bool(ref["confirmation_learning_pass"])
                for ref in study["references"].values()
            )
            summary["protection_verdict"] = {
                family: bool(result.get("protection"))
                and bool(summary["apparatus_valid"])
                and bool(diet["quality_confirmed"])
                for family, result in study["effects"].items()
            }
        if stage == "selection":
            proposed = _protocol(config, development, output_root, values, source_fp)
            protocol_path = output_root / "protocol.json"
            if protocol_path.exists() and json.loads(protocol_path.read_text()) != proposed:
                raise ValueError("registered confirmation protocol differs from new proposal")
            protocol_path.write_text(dumps_valid(proposed), encoding="utf-8")
    else:
        raise ValueError(f"unknown stage {stage}")
    (output_root / f"{stage}_summary.json").write_text(dumps_valid(summary), encoding="utf-8")
    return summary
