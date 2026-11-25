"""A small, self-contained Vision Transformer encoder (the shared backbone).

Phase 0 needs *one* tiny, CPU-friendly encoder that both SSL methods can share:

* **MAE** needs patch-level masking and access to the visible-token latent, plus an
  ``ids_restore`` permutation so a decoder can put the tokens back (see
  :class:`cafl4ds.models.heads.MAEDecoder`).
* **SimSiam** (and later BYOL/SimCLR) only needs a pooled embedding of the full image.

So :class:`TinyViTEncoder` exposes :meth:`forward_encoder` (optionally masked, the MAE
entry point) and :meth:`embed` (a pooled ``[B, d]`` representation, the joint-embedding /
measurement entry point). It is deliberately hand-rolled and tiny — no external model zoo —
so the from-scratch smoke test runs on CPU in seconds; at HPU scale a larger / pretrained
encoder satisfying the same two methods can be dropped in via config.

Shapes follow the standard ViT/MAE convention: images are ``[B, C, H, W]``; a patch
sequence is ``[B, N, d]`` with ``N = (H / p) * (W / p)`` patches; the token sequence handed
to the transformer blocks is ``[B, 1 + N, d]`` (a prepended ``cls`` token).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import isfinite, sqrt
from typing import cast

import torch
from torch import nn


@dataclass(eq=False)
class FeedForwardOutput:
    """Feedforward features, unweighted balancing loss, and full router probabilities."""

    features: torch.Tensor
    aux_loss: torch.Tensor
    routing_probs: torch.Tensor | None = None


@dataclass(eq=False)
class BlockOutput:
    """Residual block output with optional attention and routing readouts."""

    features: torch.Tensor
    aux_loss: torch.Tensor
    attention: torch.Tensor | None = None
    routing_probs: torch.Tensor | None = None


@dataclass(eq=False)
class EncoderOutput:
    """Encoder tokens and MAE metadata, with an explicit scalar auxiliary loss."""

    tokens: torch.Tensor
    mask: torch.Tensor | None
    ids_restore: torch.Tensor | None
    aux_loss: torch.Tensor
    attentions: list[torch.Tensor] = field(default_factory=list)
    routing_probs: dict[int, torch.Tensor] = field(default_factory=dict)


def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Split a batch of images into a sequence of flattened patches.

    Args:
        imgs: Images ``[B, C, H, W]`` with ``H`` and ``W`` divisible by ``patch_size``.
        patch_size: Side length ``p`` of each square patch.

    Returns:
        Patches ``[B, N, p * p * C]`` with ``N = (H / p) * (W / p)``, in row-major
        (top-left to bottom-right) patch order.
    """
    b, c, h, w = imgs.shape
    p = patch_size
    nh, nw = h // p, w // p
    x = imgs.reshape(b, c, nh, p, nw, p)
    x = torch.einsum("bchpwq->bhwpqc", x)
    return x.reshape(b, nh * nw, p * p * c)


def unpatchify(patches: torch.Tensor, patch_size: int, channels: int) -> torch.Tensor:
    """Inverse of :func:`patchify`: reassemble a patch sequence into images.

    Args:
        patches: Patches ``[B, N, p * p * C]`` in row-major patch order.
        patch_size: Side length ``p`` of each square patch.
        channels: Number of image channels ``C``.

    Returns:
        Images ``[B, C, H, W]`` with ``H = W = p * sqrt(N)`` (square grid assumed).
    """
    b, n, _ = patches.shape
    p, c = patch_size, channels
    g = int(round(n**0.5))
    x = patches.reshape(b, g, g, p, p, c)
    x = torch.einsum("bhwpqc->bchpwq", x)
    return x.reshape(b, c, g * p, g * p)


class Mlp(nn.Module):  # type: ignore[misc]  # nn.Module is Any without torch stubs (mypy hook env)
    """A two-layer MLP with GELU activation (transformer feed-forward block)."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        """Build the MLP.

        Args:
            dim: Input and output dimensionality.
            hidden_dim: Width of the hidden layer.
        """
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP to ``x`` (``[..., dim] -> [..., dim]``)."""
        return cast(torch.Tensor, self.fc2(self.act(self.fc1(x))))

    def forward_with_aux(self, x: torch.Tensor, *, deterministic: bool = False) -> FeedForwardOutput:
        """Return dense features with zero auxiliary loss; deterministic is a shared option."""
        return FeedForwardOutput(self(x), x.new_zeros(()))


def mix_all_experts(experts: nn.ModuleList, x: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    """Apply every expert with full normalized weights, without sparse gather/scatter."""
    outputs = torch.stack([expert(x) for expert in experts], dim=-2)
    return cast(torch.Tensor, (outputs * probabilities.to(outputs.dtype).unsqueeze(-1)).sum(dim=-2))


class MoEMlp(nn.Module):  # type: ignore[misc]
    """Sparse top-k experts with batch importance/load balancing.

    Softmax is computed over all experts before top-k. Selected probabilities
    weight the expert outputs without renormalization, retaining task gradients
    through the gate even for top-1.
    Router noise has standard deviation ``router_noise_std / num_experts`` during
    training, following V-MoE. Evaluation and diagnostic reads are deterministic.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int = 4,
        top_k: int = 1,
        routing_level: str = "token",
        router_noise_std: float = 1.0,
        importance_weight: float = 1.0,
        load_weight: float = 1.0,
    ) -> None:
        """Build experts and router; loss weights apply before SSL's aux_weight.

        Image routing pools all input tokens, including CLS when present. With
        zero noise, load is a hard-count diagnostic (no load gradient); importance
        remains differentiable. No expert capacity or token dropping is imposed.
        """
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("Require 1 <= top_k <= num_experts.")
        if routing_level not in {"token", "image"}:
            raise ValueError("routing_level must be 'token' or 'image'.")
        for name, value in (
            ("router_noise_std", router_noise_std),
            ("importance_weight", importance_weight),
            ("load_weight", load_weight),
        ):
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        self.experts = nn.ModuleList([Mlp(dim, hidden_dim) for _ in range(num_experts)])
        self.router = nn.Linear(dim, num_experts)
        self.num_experts = num_experts
        self.top_k = top_k
        self.routing_level = routing_level
        self.router_noise_std = router_noise_std
        self.importance_weight = importance_weight
        self.load_weight = load_weight

    @staticmethod
    def _cv_squared(values: torch.Tensor) -> torch.Tensor:
        """Population coefficient of variation squared, zero for uniform usage."""
        return values.var(unbiased=False) / values.mean().square().clamp_min(1e-9)

    def _balancing_loss(self, logits: torch.Tensor, routed_logits: torch.Tensor, noise_std: float) -> torch.Tensor:
        """V-MoE importance and expected-load losses over this batch's routing units.

        Importance uses full clean softmax probabilities. Expected load is the
        Gaussian probability of exceeding the noisy K-th logit (V-MoE Appendix
        A.2), rather than a nondifferentiable count of selected experts. Image
        decisions are counted once per image, before broadcasting over tokens.
        """
        importance = logits.softmax(dim=-1).sum(dim=0)
        importance_loss = self._cv_squared(importance)
        if self.top_k == self.num_experts:
            load_loss = logits.new_zeros(())
        elif noise_std > 0:
            threshold = routed_logits.topk(self.top_k, dim=-1).values[:, -1:]
            expected_load = (0.5 * torch.erfc((threshold - logits) / (noise_std * sqrt(2)))).mean(dim=0)
            load_loss = self._cv_squared(expected_load)
        else:
            ids = routed_logits.topk(self.top_k, dim=-1).indices
            load = torch.nn.functional.one_hot(ids, self.num_experts).float().sum(dim=(0, 1))
            load_loss = self._cv_squared(load)
        return self.importance_weight * importance_loss + self.load_weight * load_loss

    def forward(self, x: torch.Tensor, *, deterministic: bool = False) -> FeedForwardOutput:
        """Route [B, N, D] tokens and return features, loss, and [B, N, E] probabilities."""
        b, n, d = x.shape
        router_input = x.mean(dim=1) if self.routing_level == "image" else x
        logits = self.router(router_input).float().reshape(-1, self.num_experts)
        noise_std = self.router_noise_std / self.num_experts if self.training and not deterministic else 0.0
        routed_logits = logits + noise_std * torch.randn_like(logits) if noise_std > 0 else logits
        probs = routed_logits.softmax(dim=-1)
        weights, expert_ids = probs.topk(self.top_k, dim=-1)
        aux_loss = self._balancing_loss(logits, routed_logits, noise_std)
        if self.routing_level == "image":
            expert_ids = expert_ids[:, None, :].expand(-1, n, -1)
            weights = weights[:, None, :].expand(-1, n, -1)
            probs = probs[:, None, :].expand(-1, n, -1)
        if self.top_k == self.num_experts:
            full_probs = probs.reshape(b, n, self.num_experts)
            return FeedForwardOutput(mix_all_experts(self.experts, x, full_probs), aux_loss, full_probs)
        tokens = x.reshape(b * n, d)
        expert_ids = expert_ids.reshape(b * n, self.top_k)
        weights = weights.reshape(b * n, self.top_k)
        out = torch.zeros_like(tokens)
        for expert_id, expert in enumerate(self.experts):
            token_idx, slot_idx = torch.where(expert_ids == expert_id)
            if token_idx.numel() == 0:
                continue
            expert_out = expert(tokens[token_idx])
            weighted = expert_out * weights[token_idx, slot_idx, None]
            out.index_add_(0, token_idx, weighted.to(out.dtype))
        return FeedForwardOutput(out.reshape(b, n, d), aux_loss, probs.reshape(b, n, self.num_experts))

    def forward_with_aux(self, x: torch.Tensor, *, deterministic: bool = False) -> FeedForwardOutput:
        """Use the same explicit-result interface as the dense feedforward layer."""
        return cast(FeedForwardOutput, self(x, deterministic=deterministic))


class Attention(nn.Module):  # type: ignore[misc]  # nn.Module is Any without torch stubs (mypy hook env)
    """Standard multi-head self-attention over a token sequence."""

    def __init__(self, dim: int, num_heads: int) -> None:
        """Build the attention layer.

        Args:
            dim: Token dimensionality; must be divisible by ``num_heads``.
            num_heads: Number of attention heads.

        Raises:
            ValueError: If ``dim`` is not divisible by ``num_heads``.
        """
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Self-attend over the token dimension (``[B, T, d] -> [B, T, d]``).

        The default path uses the fused ``scaled_dot_product_attention`` (fast; the training
        path). When ``return_attn`` is set, the attention probabilities are computed explicitly
        and returned alongside the output — a **read-only** path for the mean-attention-distance
        instrument (P0.5.2), never used in training.

        Args:
            x: Token sequence ``[B, T, d]``.
            return_attn: If ``True``, also return the ``[B, num_heads, T, T]`` attention
                probability matrix.

        Returns:
            The attended tokens ``[B, T, d]``, or — when ``return_attn`` — a
            ``(tokens, attn)`` pair.
        """
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if return_attn:
            attn = torch.softmax((q @ k.transpose(-2, -1)) * (self.head_dim**-0.5), dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(b, t, d)
            return self.proj(out), attn
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, t, d)
        return cast(torch.Tensor, self.proj(out))


class Block(nn.Module):  # type: ignore[misc]  # nn.Module is Any without torch stubs (mypy hook env)
    """A pre-norm transformer block: attention + MLP with residual connections."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, mlp: Mlp | MoEMlp | None = None) -> None:
        """Build the transformer block.

        Args:
            dim: Token dimensionality.
            num_heads: Number of attention heads.
            mlp_ratio: Hidden-layer width of the MLP as a multiple of ``dim``.
            mlp: Optional feedforward layer; defaults to the original dense MLP.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = mlp if mlp is not None else Mlp(dim, int(dim * mlp_ratio))

    def forward(
        self, x: torch.Tensor, *, return_aux: bool = False, return_attn: bool = False, deterministic: bool = False
    ) -> torch.Tensor | BlockOutput:
        """Apply attention then MLP, each with a residual connection."""
        attention = None
        if return_attn:
            attended, attention = self.attn(self.norm1(x), return_attn=True)
        else:
            attended = self.attn(self.norm1(x))
        x = x + attended
        result = self.mlp.forward_with_aux(self.norm2(x), deterministic=deterministic)
        x = x + result.features
        if return_aux:
            return BlockOutput(x, result.aux_loss, attention, result.routing_probs)
        return x

    def forward_with_aux(
        self, x: torch.Tensor, *, return_attn: bool = False, deterministic: bool = False
    ) -> BlockOutput:
        """Return features and auxiliary loss using the normal module call path."""
        return cast(BlockOutput, self(x, return_aux=True, return_attn=return_attn, deterministic=deterministic))


class TinyViTEncoder(nn.Module):  # type: ignore[misc]  # nn.Module is Any without torch stubs (mypy hook env)
    """A tiny ViT encoder shared by the SSL methods.

    Exposes two entry points:

    * :meth:`forward_encoder` — run the transformer over the (optionally masked) patch
      sequence; the MAE path uses ``mask_ratio > 0`` and consumes the returned mask /
      restore indices.
    * :meth:`embed` — a pooled ``[B, embed_dim]`` representation of the full image (mean of
      the patch tokens after the final norm); the joint-embedding and measurement path.

    The embedding read by the health instruments and probes is deliberately the *backbone*
    representation (:meth:`embed`), never a projector/predictor output — those are training
    heads, not the representation under study.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 8,
        in_chans: int = 3,
        embed_dim: int = 96,
        depth: int = 4,
        num_heads: int = 3,
        mlp_ratio: float = 2.0,
        *,
        block_factory: Callable[[int], Block] | None = None,
    ) -> None:
        """Build the encoder.

        Args:
            img_size: Input image side length (square); must be divisible by ``patch_size``.
            patch_size: Side length of each square patch.
            in_chans: Number of input channels.
            embed_dim: Token / embedding dimensionality.
            depth: Number of transformer blocks.
            num_heads: Number of attention heads per block.
            mlp_ratio: MLP hidden width as a multiple of ``embed_dim``.
            block_factory: Optional layer-index factory, called before weight initialization.

        Raises:
            ValueError: If ``img_size`` is not divisible by ``patch_size``.
        """
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size ({img_size}) must be divisible by patch_size ({patch_size}).")
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.blocks = nn.ModuleList(
            [block_factory(i) if block_factory else Block(embed_dim, num_heads, mlp_ratio) for i in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize positional/token parameters and the linear/conv layers."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def reset_parameters_from_scratch(self) -> None:
        """Re-randomize *all* learned weights, discarding any loaded checkpoint.

        Builds a genuine random-init reference from a subclass that warm-starts in ``__init__``
        (e.g. the P0.3.7 savings ``scratch`` baseline from :class:`MAEPretrainedViTEncoder`): it
        reruns :meth:`_init_weights` (positional/token params, Linear, LayerNorm) *and* resets the
        patch-embed conv (which ``_init_weights`` leaves at its construction default), while leaving
        registered buffers — e.g. a subclass's normalization constants — untouched, so the encoder's
        input contract is preserved and only the pretrained knowledge is wiped.
        """
        self._init_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()

    def _embed_patches(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convolutional patch embedding, adding the patch positional encodings.

        Args:
            imgs: Images ``[B, C, H, W]``.

        Returns:
            Patch tokens ``[B, N, embed_dim]`` with positional embeddings added (the ``cls``
            token and its position are handled by the callers).
        """
        x = self.patch_embed(imgs).flatten(2).transpose(1, 2)  # [B, N, D]
        return cast(torch.Tensor, x + self.pos_embed[:, 1:, :])

    @staticmethod
    def random_masking(
        x: torch.Tensor, mask_ratio: float, *, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-sample random masking of a patch sequence (MAE, He et al. 2022).

        Args:
            x: Patch tokens ``[B, N, d]``.
            mask_ratio: Fraction of patches to drop, in ``[0, 1)``.
            noise: Optional fixed ``[B, N]`` mask priorities for paired experiments.

        Returns:
            A tuple ``(x_kept, mask, ids_restore)`` where ``x_kept`` is ``[B, N_keep, d]``
            (the visible tokens), ``mask`` is ``[B, N]`` (1 = masked/removed, 0 = kept), and
            ``ids_restore`` is ``[B, N]`` — the permutation that undoes the shuffle.
        """
        b, n, d = x.shape
        len_keep = max(1, int(round(n * (1.0 - mask_ratio))))
        if noise is None:
            noise = torch.rand(b, n, device=x.device)
        elif noise.shape != (b, n):
            raise ValueError("Mask noise must have shape [B, N].")
        else:
            noise = noise.to(x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, d))
        mask = torch.ones(b, n, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_kept, mask, ids_restore

    def forward_encoder(
        self, imgs: torch.Tensor, mask_ratio: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Run the transformer over the (optionally masked) patch sequence.

        Args:
            imgs: Images ``[B, C, H, W]``.
            mask_ratio: Fraction of patches to mask before encoding (MAE). ``0`` disables
                masking and processes the full sequence (joint-embedding / measurement).

        Returns:
            A tuple ``(tokens, mask, ids_restore)``. ``tokens`` is ``[B, 1 + N_keep, d]``
            (final-normed, ``cls`` token first). ``mask`` and ``ids_restore`` are ``[B, N]``
            when ``mask_ratio > 0`` and ``None`` otherwise.
        """
        result = self.forward_with_aux(imgs, mask_ratio)
        return result.tokens, result.mask, result.ids_restore

    def _encode(
        self,
        imgs: torch.Tensor,
        mask_ratio: float = 0.0,
        *,
        collect_attn: bool = False,
        collect_routing: bool = False,
        deterministic: bool = False,
        mask_noise: torch.Tensor | None = None,
    ) -> EncoderOutput:
        """Shared token preparation and block execution for training and diagnostics."""
        x = self._embed_patches(imgs)
        mask: torch.Tensor | None = None
        ids_restore: torch.Tensor | None = None
        if mask_ratio > 0.0:
            if mask_noise is None:
                x, mask, ids_restore = self.random_masking(x, mask_ratio)
            else:
                x, mask, ids_restore = self.random_masking(x, mask_ratio, noise=mask_noise)
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        losses: list[torch.Tensor] = []
        attentions: list[torch.Tensor] = []
        routing_probs: dict[int, torch.Tensor] = {}
        for index, blk in enumerate(self.blocks):
            blk = cast(Block, blk)
            result = blk.forward_with_aux(x, return_attn=collect_attn, deterministic=deterministic)
            x = result.features
            if result.routing_probs is not None:
                losses.append(result.aux_loss)
                if collect_routing:
                    routing_probs[index] = result.routing_probs
            if result.attention is not None:
                attentions.append(result.attention)
        aux_loss = torch.stack(losses).mean() if losses else x.new_zeros(())
        return EncoderOutput(self.norm(x), mask, ids_restore, aux_loss, attentions, routing_probs)

    def forward_with_aux(
        self, imgs: torch.Tensor, mask_ratio: float = 0.0, *, mask_noise: torch.Tensor | None = None
    ) -> EncoderOutput:
        """Return tokens, MAE metadata, and mean balancing loss over MoE layers.

        Dense layers contribute no auxiliary loss. The returned loss belongs to
        this call's graph; no last-forward state is stored on the encoder.
        """
        return self._encode(imgs, mask_ratio, mask_noise=mask_noise)

    def embed_with_aux(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pooled patch features and auxiliary loss from a single forward."""
        result = self.forward_with_aux(imgs.to(self.pos_embed.device))
        return result.tokens[:, 1:, :].mean(dim=1), result.aux_loss

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Return the full-sequence tokens ``[B, 1 + N, d]`` (unmasked)."""
        tokens, _, _ = self.forward_encoder(imgs, mask_ratio=0.0)
        return tokens

    def embed(self, imgs: torch.Tensor) -> torch.Tensor:
        """Return a pooled ``[B, embed_dim]`` representation (mean of the patch tokens).

        This is the representation consumed by the health instruments and probes — the
        backbone output, never a training-head projection.

        Args:
            imgs: Images ``[B, C, H, W]``.

        Returns:
            The mean-pooled patch-token embedding ``[B, embed_dim]``.
        """
        # The measurement/probe callers hold their fixed eval tensors on CPU, so coerce to
        # the backbone's device here (a no-op when both are CPU). This keeps ``embed`` — the
        # instrument entry point — usable when the model has been moved to an accelerator
        # (e.g. ``hpu``) without pushing device bookkeeping into the monitor/measurements.
        imgs = imgs.to(self.pos_embed.device)
        features, _ = self.embed_with_aux(imgs)
        return features

    @property
    def grid_size(self) -> int:
        """Side length of the square patch grid (``num_patches = grid_size**2``)."""
        return int(round(self.num_patches**0.5))

    @torch.no_grad()  # type: ignore[misc]  # untyped decorator without torch stubs (mypy hook env)
    def attention_maps(self, imgs: torch.Tensor) -> list[torch.Tensor]:
        """Return the per-block self-attention probability matrices (read-only, unmasked).

        Mirrors the pre-norm block computation of :meth:`forward` exactly (so the captured
        attention is the model's true attention), but threads ``return_attn`` through each block's
        attention to collect the ``[B, num_heads, T, T]`` probability matrix per block. Used by the
        mean-attention-distance instrument (P0.5.2); it never enters training.

        Args:
            imgs: Images ``[B, C, H, W]``.

        Returns:
            A list of ``depth`` attention tensors, each ``[B, num_heads, 1 + N, 1 + N]``.
        """
        imgs = imgs.to(self.pos_embed.device)
        return self._encode(imgs, collect_attn=True, deterministic=True).attentions


class MoEViTEncoder(TinyViTEncoder):
    """Tiny ViT with sparse experts in selected blocks and explicit auxiliary losses.

    Existing forward/forward_encoder/embed methods retain their dense interfaces.
    Train with forward_with_aux or embed_with_aux to include batch balancing.
    Block indices are zero-based; by default the last two blocks are converted.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 8,
        in_chans: int = 3,
        embed_dim: int = 96,
        depth: int = 4,
        num_heads: int = 3,
        mlp_ratio: float = 2.0,
        num_experts: int = 4,
        top_k: int = 1,
        moe_blocks: Sequence[int] | None = None,
        routing_level: str = "token",
        balancing_mode: str = "batch",
        router_noise_std: float = 1.0,
        importance_weight: float = 1.0,
        load_weight: float = 1.0,
    ) -> None:
        """Construct selected MoE blocks before the inherited initialization pass.

        Only batch balancing is supported. Set both balancing weights to zero
        to disable it; EMA, bias balancing, and consistency are not implemented.
        """
        if depth < 1:
            raise ValueError("depth must be positive.")
        indices = tuple(range(max(0, depth - 2), depth)) if moe_blocks is None else tuple(moe_blocks)
        if len(set(indices)) != len(indices) or any(not isinstance(i, int) or i < 0 or i >= depth for i in indices):
            raise ValueError("moe_blocks must contain unique zero-based indices within depth.")
        if balancing_mode != "batch":
            raise ValueError("Only balancing_mode='batch' is supported.")
        # Validate options even when the caller requests no MoE blocks.
        if not 1 <= top_k <= num_experts:
            raise ValueError("Require 1 <= top_k <= num_experts.")
        if routing_level not in {"token", "image"}:
            raise ValueError("routing_level must be 'token' or 'image'.")

        def make_block(index: int) -> Block:
            mlp = (
                MoEMlp(
                    embed_dim,
                    int(embed_dim * mlp_ratio),
                    num_experts,
                    top_k,
                    routing_level,
                    router_noise_std,
                    importance_weight,
                    load_weight,
                )
                if index in indices
                else None
            )
            return Block(embed_dim, num_heads, mlp_ratio, mlp=mlp)

        super().__init__(
            img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, block_factory=make_block
        )
        self.moe_blocks = tuple(sorted(indices))

    @torch.no_grad()  # type: ignore[misc]
    def routing(self, imgs: torch.Tensor) -> dict[int, torch.Tensor]:
        """Read full clean probabilities [B, 1+N, E], keyed by zero-based block index.

        Uses full unmasked images, sequential block execution, and no router
        noise, even in train mode. Image routing broadcasts one distribution per
        image to every token. Does not change modes, RNG, parameters, or gradients.
        """
        imgs = imgs.to(self.pos_embed.device)
        return self._encode(imgs, collect_routing=True, deterministic=True).routing_probs
