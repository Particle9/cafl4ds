"""Attribute-carrying data sources for the P1.0.2 deployment-prototype diet.

Where a Phase-0 :class:`~cafl4ds.data.sources.DataSource` yields ``(images, labels)`` on a *single*
class axis, the deployment diet needs **two** axes over the same images:

* an **era-key** axis — the attribute the diet *orders on* (BDD's driving regime, ``(timeofday,
  weather)``), whose contiguous blocks become the correlated, nonstationary eras; and
* a **canary** axis — an attribute *orthogonal* to the ordering (BDD's ``scene``), used only to
  label the held-out probe set so the labelled canary reads something the diet did not sort by.

An :class:`AttributeSource` therefore returns :class:`AttributedImages` (images + both integer axes +
the default regime shift-walk order). Two concrete sources exist: :class:`BDD100KSource` over the real
BDD100K *images* + attributes, and :class:`SyntheticAttributeSource`, a network-free stand-in that
lets the whole rig — and its Tier-A wiring check — run from a fresh clone with no dataset. Images are
``float32`` ``[N, C, H, W]`` in ``[0, 1]``; both axes are integer ``[N]``.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812 - conventional alias
from loguru import logger
from PIL import Image
from torchvision import transforms

# Priority ranks that define the default BDD shift-walk: daytime before dusk before night, and clear
# before progressively adverse weather. A regime's walk position is (timeofday_rank, weather_rank), so
# the default stream visits `daytime·clear → … → night·foggy` — a nonstationary drive. Unknown values
# sort last (rank = large) rather than crashing.
_TIMEOFDAY_RANK = {"daytime": 0, "dawn/dusk": 1, "night": 2}
_WEATHER_RANK = {"clear": 0, "partly cloudy": 1, "overcast": 2, "rainy": 3, "snowy": 4, "foggy": 5}
_UNKNOWN_ATTRS = frozenset({"undefined", "", None})


@dataclass(frozen=True)
class AttributedImages:
    """Loaded images with the diet's two attribute axes and the default shift-walk order.

    Attributes:
        images: Image batch ``[N, C, H, W]`` (``float32`` in ``[0, 1]``).
        era_key: Integer regime id per image ``[N]`` — the axis the diet orders eras on.
        canary: Integer canary label per image ``[N]`` — the orthogonal probe axis.
        regime_order: Regime ids in the default shift-walk order (a stream may override it).
        regime_names: Human-readable name per regime id (e.g. ``"daytime·clear"``).
        canary_names: Human-readable name per canary id (e.g. ``"city street"``).
        object_categories: Optional per-image detection-category counts (e.g. ``{"car": 4,
            "person": 1}``) aligned to ``images`` — the free label aggregate BDD's boxes give, used
            to tag each corpus leg. ``None`` when the source carries no detection labels (synthetic).
    """

    images: torch.Tensor
    era_key: torch.Tensor
    canary: torch.Tensor
    regime_order: list[int]
    regime_names: dict[int, str]
    canary_names: dict[int, str]
    object_categories: list[dict[str, int]] | None = None


class AttributeSource(ABC):
    """Produces :class:`AttributedImages` for the regime stream to order into a correlated diet."""

    @abstractmethod
    def load(self) -> AttributedImages:
        """Load the full dataset with both attribute axes into memory."""

    @property
    @abstractmethod
    def num_canary_classes(self) -> int:
        """Number of distinct canary (probe-axis) classes."""


class SyntheticAttributeSource(AttributeSource):
    """Network-free attributed images: canary-clustered patterns on an independent regime axis.

    Each image's *content* is determined by its canary class (a distinct random pattern → the probe
    is learnable and effective rank is meaningful), plus a small per-regime tint so the regimes are
    genuinely different distributions (so representation drift responds to the shift-walk). The regime
    axis is assigned **independently** of the canary, so eras are not trivially separable by the
    canary the probe reads — exactly the orthogonality the real diet has. Regimes can be made
    long-tailed to mirror rare driving conditions.
    """

    def __init__(
        self,
        num_regimes: int = 4,
        num_canary_classes: int = 4,
        per_cell: int = 24,
        img_size: int = 16,
        channels: int = 3,
        noise: float = 0.3,
        regime_tint: float = 0.15,
        long_tail: bool = True,
        seed: int = 0,
    ) -> None:
        """Configure the synthetic attributed source.

        Args:
            num_regimes: Number of era-key regimes (the ordering axis).
            num_canary_classes: Number of canary (probe-axis) classes.
            per_cell: Base images per ``(regime, canary)`` cell before long-tail thinning.
            img_size: Image side length.
            channels: Number of channels.
            noise: Per-pixel Gaussian noise around each canary pattern.
            regime_tint: Magnitude of the per-regime additive tint (0 disables the shift).
            long_tail: If set, later regimes get geometrically fewer images (rare conditions).
            seed: RNG seed for reproducibility.
        """
        self._num_regimes = num_regimes
        self._num_canary = num_canary_classes
        self.per_cell = per_cell
        self.img_size = img_size
        self.channels = channels
        self.noise = noise
        self.regime_tint = regime_tint
        self.long_tail = long_tail
        self.seed = seed
        self._cache: AttributedImages | None = None

    @property
    def num_canary_classes(self) -> int:
        """The configured canary-class count."""
        return self._num_canary

    def load(self) -> AttributedImages:
        """Generate the attributed images.

        Returns:
            :class:`AttributedImages` with canary-clustered content, an independent (optionally
            long-tailed) regime axis, and ``regime_order = 0..num_regimes-1``. Cached on the
            instance, so threading one source through several streams (the live + gate arms) decodes
            it once.
        """
        if self._cache is not None:
            return self._cache
        g = torch.Generator().manual_seed(self.seed)
        shape = (self.channels, self.img_size, self.img_size)
        canary_patterns = [torch.rand(shape, generator=g) for _ in range(self._num_canary)]
        regime_tints = [self.regime_tint * torch.rand(shape, generator=g) for _ in range(self._num_regimes)]
        images, era_key, canary = [], [], []
        for r in range(self._num_regimes):
            # Long-tail: regime r keeps ~per_cell / 2**r of the base count (at least 1 per canary).
            keep = max(1, self.per_cell // (2**r)) if self.long_tail else self.per_cell
            for k in range(self._num_canary):
                base = canary_patterns[k].unsqueeze(0) + regime_tints[r].unsqueeze(0)
                block = base + self.noise * torch.randn(keep, *shape, generator=g)
                images.append(block.clamp_(0.0, 1.0))
                era_key.append(torch.full((keep,), r, dtype=torch.long))
                canary.append(torch.full((keep,), k, dtype=torch.long))
        self._cache = AttributedImages(
            images=torch.cat(images),
            era_key=torch.cat(era_key),
            canary=torch.cat(canary),
            regime_order=list(range(self._num_regimes)),
            regime_names={r: f"regime{r}" for r in range(self._num_regimes)},
            canary_names={k: f"canary{k}" for k in range(self._num_canary)},
        )
        return self._cache


class BDD100KSource(AttributeSource):
    """The real BDD100K *images* with their driving attributes — the deployment substrate.

    Reads the BDD100K 100k-image set and the per-image attribute records (``weather`` / ``scene`` /
    ``timeofday``), mapping ``(timeofday, weather)`` to the **era-key** regime axis and ``scene`` to
    the orthogonal **canary** axis. Regime ids are assigned in shift-walk order (see the module
    ranks), so ``regime_order`` is simply ``0..R-1`` and lower ids are earlier-in-the-drive regimes.
    Images with an undefined attribute on either axis are skipped. Native BDD frames are 1280x720, so
    they are resized to ``img_size`` (also the low-memory portability lever). Video is *not* used.

    Images live under ``images/100k/<split>/*.jpg``. With ``legacy_labels=True`` (the default),
    labels come from ``labels/bdd100k_labels_images_<split>.json``. With ``legacy_labels=False``,
    labels come from per-image JSON files in ``labels/100k/<split>/`` — both paths are overridable.
    """

    def __init__(
        self,
        bdd_root: str,
        split: str = "train",
        img_size: int = 128,
        max_images: int | None = None,
        images_dir: str | None = None,
        labels_file: str | None = None,
        min_canary_count: int = 0,
        *,
        legacy_labels: bool = True,
    ) -> None:
        """Configure the BDD100K source.

        Args:
            bdd_root: Root of the downloaded BDD100K distribution.
            split: Which split to load (``"train"`` or ``"val"``).
            img_size: Side length to resize the native 1280x720 frames to.
            max_images: If set, keep at most this many (attribute-valid) images.
            images_dir: Override for the image directory (default ``<root>/images/100k/<split>``).
            labels_file: Override for the attributes JSON file in legacy mode (default
                ``<root>/labels/bdd100k_labels_images_<split>.json``), or the per-image JSON
                directory when ``legacy_labels=False`` (default ``<root>/labels/100k/<split>``).
            min_canary_count: Drop images whose **scene** (canary class) has fewer than this many
                attribute-valid images. Real BDD has a long scene tail (e.g. ``tunnel`` ≈ a handful of
                frames) that cannot support a balanced held-out probe; this keeps the canary to
                probeable classes. ``0`` (default) keeps every observed scene — no change.
            legacy_labels: Read the combined legacy JSON array when true. When false, read
                per-image records with top-level attributes and detections in ``frames[0].objects``.
        """
        self.bdd_root = bdd_root
        self.split = split
        self.img_size = img_size
        self.max_images = max_images
        self._images_dir = images_dir
        self._labels_file = labels_file
        self.min_canary_count = min_canary_count
        self.legacy_labels = legacy_labels
        self._num_canary = 0  # set on load (number of observed scenes)
        self._cache: AttributedImages | None = None

    @property
    def num_canary_classes(self) -> int:
        """Number of distinct scenes observed (populated by :meth:`load`)."""
        return self._num_canary

    def _paths(self) -> tuple[Path, Path]:
        """Resolve the image directory and labels file or directory, honouring the overrides."""
        images_dir = (
            Path(self._images_dir) if self._images_dir else Path(self.bdd_root) / "images" / "100k" / self.split
        )
        labels_file = (
            Path(self._labels_file)
            if self._labels_file
            else (
                Path(self.bdd_root) / "labels" / f"bdd100k_labels_images_{self.split}.json"
                if self.legacy_labels
                else Path(self.bdd_root) / "labels" / "100k" / self.split
            )
        )
        return images_dir, labels_file

    def _records(self, labels_path: Path) -> Iterator[dict[str, Any]]:
        """Yield legacy-shaped records; per-image files are read lazily in filename order."""
        if self.legacy_labels:
            if not labels_path.is_file():
                raise FileNotFoundError(
                    f"BDD100K attributes JSON not found at {labels_path}. Download the labels and place them "
                    "under <bdd_root>/labels/ (or pass labels_file=)."
                )
            yield from json.loads(labels_path.read_text(encoding="utf-8"))
            return
        if not labels_path.is_dir():
            raise FileNotFoundError(
                f"BDD100K per-image labels not found at {labels_path}. Place the JSON files under "
                "<bdd_root>/labels/100k/<split>/ (or pass labels_file= with a directory)."
            )
        for label_file in sorted(labels_path.glob("*.json")):
            record = json.loads(label_file.read_text(encoding="utf-8"))
            name = record["name"]
            frames = record.get("frames", [])
            yield {
                "name": name if name.endswith(".jpg") else f"{name}.jpg",
                "attributes": record.get("attributes", {}),
                "labels": frames[0].get("objects", []) if frames else [],
            }

    def load(self) -> AttributedImages:
        """Parse the attribute records, decode the referenced images, and build both axes.

        Returns:
            :class:`AttributedImages` with regimes id'd in shift-walk order and scenes as the canary.

        The decoded result is cached on the instance: threading one source through the live and
        (optional) gate arms then decodes the corpus **once** rather than 2–3× per drive.

        Raises:
            FileNotFoundError: If the image directory or the labels file/directory is missing.
            ValueError: If no image survives attribute filtering.
        """
        if self._cache is None:
            self._cache = self._decode()
        return self._cache

    def _decode(self) -> AttributedImages:
        """Parse the attribute records and decode the referenced images (the uncached work)."""
        images_dir, labels_file = self._paths()
        if not images_dir.is_dir():
            raise FileNotFoundError(
                f"BDD100K images not found at {images_dir}. Download the 100k images and arrange them "
                "under <bdd_root>/images/100k/<split>/ (see docs/experiments/phase1/P1.0.2.md)."
            )
        # Collect (path, regime-tuple, scene, category-counts) for every attribute-valid, present image.
        valid: list[tuple[Path, tuple[str, str], str, dict[str, int]]] = []
        for rec in self._records(labels_file):
            attrs = rec.get("attributes", {})
            timeofday, weather, scene = attrs.get("timeofday"), attrs.get("weather"), attrs.get("scene")
            if timeofday in _UNKNOWN_ATTRS or weather in _UNKNOWN_ATTRS or scene in _UNKNOWN_ATTRS:
                continue
            path = images_dir / rec["name"]
            if not path.is_file():
                continue
            valid.append((path, (timeofday, weather), scene, _count_categories(rec)))
            if self.max_images is not None and len(valid) >= self.max_images:
                break
        if not valid:
            raise ValueError(f"no attribute-valid BDD100K images found under {images_dir}")

        if self.min_canary_count > 0:
            valid = _drop_rare_scenes(valid, self.min_canary_count)

        regime_order, regime_id, regime_names = _rank_regimes({r for _, r, _, _ in valid})
        scene_id, canary_names = _index_scenes({s for _, _, s, _ in valid})
        self._num_canary = len(scene_id)

        resize = transforms.Compose([transforms.Resize((self.img_size, self.img_size)), transforms.ToTensor()])
        imgs, era_key, canary, categories = [], [], [], []
        for path, regime, scene, cats in valid:
            with Image.open(path) as im:
                imgs.append(resize(im.convert("RGB")))
            era_key.append(regime_id[regime])
            canary.append(scene_id[scene])
            categories.append(cats)
        logger.info(
            f"BDD100KSource: loaded {len(imgs)} images ({self.split}) at {self.img_size}px "
            f"over {len(regime_order)} regimes, {len(scene_id)} scenes"
        )
        return AttributedImages(
            images=torch.stack(imgs),
            era_key=torch.tensor(era_key, dtype=torch.long),
            canary=torch.tensor(canary, dtype=torch.long),
            regime_order=regime_order,
            regime_names=regime_names,
            canary_names=canary_names,
            object_categories=categories,
        )


def _rank_regimes(regimes: set[tuple[str, str]]) -> tuple[list[int], dict[tuple[str, str], int], dict[int, str]]:
    """Assign regime ids in shift-walk order, so ``regime_order`` is ``0..R-1``.

    Args:
        regimes: The observed ``(timeofday, weather)`` combinations.

    Returns:
        ``(regime_order, regime_id, regime_names)`` — the id list in walk order, the tuple→id map, and
        the id→name map. Ids follow ``(timeofday_rank, weather_rank)``, so id 0 is the earliest regime.
    """
    ordered = sorted(regimes, key=lambda tw: (_TIMEOFDAY_RANK.get(tw[0], 99), _WEATHER_RANK.get(tw[1], 99), tw))
    regime_id = {tw: i for i, tw in enumerate(ordered)}
    regime_names = {i: f"{tw[0]}·{tw[1]}" for tw, i in regime_id.items()}
    return list(range(len(ordered))), regime_id, regime_names


def _count_categories(record: dict[str, object]) -> dict[str, int]:
    """Count detection-box categories in a BDD label record (the free per-image object aggregate).

    Args:
        record: One BDD100K label record (its ``labels`` list holds the detection boxes, each with a
            ``category``); a record with no boxes yields an empty count.

    Returns:
        A ``category -> count`` map over the record's boxes.
    """
    counts: dict[str, int] = {}
    labels = record.get("labels")
    if isinstance(labels, list):
        for box in labels:
            category = box.get("category") if isinstance(box, dict) else None
            if isinstance(category, str):
                counts[category] = counts.get(category, 0) + 1
    return counts


def _drop_rare_scenes(
    valid: list[tuple[Path, tuple[str, str], str, dict[str, int]]], min_count: int
) -> list[tuple[Path, tuple[str, str], str, dict[str, int]]]:
    """Drop images whose scene (canary class) has fewer than ``min_count`` valid images.

    BDD's scene axis is long-tailed — rare scenes (``tunnel``, ``gas stations``) carry too few frames
    to reserve a balanced held-out probe. This trims those from the canary; the ordering (regime) axis
    is untouched beyond losing those few images.

    Args:
        valid: The collected ``(path, regime, scene, category-counts)`` tuples.
        min_count: The per-scene frame floor.

    Returns:
        The subset whose scene is populated enough to probe.

    Raises:
        ValueError: If no scene meets the floor.
    """
    scene_counts: dict[str, int] = {}
    for _, _, scene, _ in valid:
        scene_counts[scene] = scene_counts.get(scene, 0) + 1
    kept = {s for s, n in scene_counts.items() if n >= min_count}
    if not kept:
        raise ValueError(f"no BDD100K scene has >= {min_count} images (min_canary_count too high)")
    dropped = sorted(set(scene_counts) - kept)
    if dropped:
        logger.info(f"BDD100KSource: dropped rare scenes below min_canary_count={min_count}: {dropped}")
    return [v for v in valid if v[2] in kept]


def _index_scenes(scenes: set[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Assign contiguous canary ids to the observed scenes (sorted for determinism).

    Args:
        scenes: The observed scene names.

    Returns:
        ``(scene_id, canary_names)`` — the name→id map and the id→name map.
    """
    ordered = sorted(scenes)
    scene_id = {s: i for i, s in enumerate(ordered)}
    return scene_id, {i: s for s, i in scene_id.items()}


def resize_images(images: torch.Tensor, img_size: int) -> torch.Tensor:
    """Bilinearly resize ``[N, C, H, W]`` images to ``img_size`` (the memory / portability lever).

    Args:
        images: Images ``[N, C, H, W]`` in ``[0, 1]``.
        img_size: Target side length.

    Returns:
        The resized images ``[N, C, img_size, img_size]``.
    """
    if images.shape[-1] == img_size and images.shape[-2] == img_size:
        return images
    return F.interpolate(images, size=img_size, mode="bilinear", align_corners=False, antialias=True)
