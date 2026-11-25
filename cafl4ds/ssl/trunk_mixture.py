"""Image-conditioned MAE mixtures with private ViT trunks."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch
from torch import nn

from cafl4ds.models.heads import MAEDecoder
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.ssl.mae import MAE


class TrunkMixtureMAE(MAE):
    """Share a prefix or mix full encoders, optionally after private decoding."""

    def __init__(
        self,
        encoder: TinyViTEncoder,
        decoder: MAEDecoder,
        branches: Sequence[nn.Module],
        additional_decoders: Sequence[MAEDecoder],
        *,
        shared_prefix: bool,
        mask_ratio: float,
        norm_pix_loss: bool,
    ) -> None:
        """Register each private branch and decoder exactly once."""
        super().__init__(encoder, decoder, mask_ratio, norm_pix_loss, aux_weight=0.0)
        self.branches = nn.ModuleList(branches)
        self.additional_decoders = nn.ModuleList(additional_decoders)
        self.shared_prefix = shared_prefix
        self.probabilities: torch.Tensor | None = None
        if len(branches) + int(not shared_prefix) != 4:
            raise ValueError("Expected four expert branches.")
        if additional_decoders and (shared_prefix or len(additional_decoders) != 3):
            raise ValueError("Private decoding requires one decoder for each full trunk.")

    def expert_modules(self) -> list[tuple[nn.Module, ...]]:
        """Return one optimizer ownership group per expert."""
        encoders = list(self.branches) if self.shared_prefix else [self.encoder, *self.branches]
        if self.additional_decoders:
            return list(zip(encoders, [self.decoder, *self.additional_decoders], strict=True))
        return [(encoder,) for encoder in encoders]

    @contextmanager
    def image_routing(self, probabilities: torch.Tensor) -> Iterator[None]:
        """Bind the detached GMM posterior to a single batch."""
        if probabilities.ndim != 2 or probabilities.shape[1] != 4:
            raise ValueError("Expected image-by-expert probabilities.")
        sums = probabilities.sum(-1, keepdim=True)
        if not bool((torch.isfinite(probabilities) & (probabilities >= 0)).all()) or not bool((sums > 0).all()):
            raise ValueError("Expected finite, nonnegative probabilities and positive mass per image.")
        previous = self.probabilities
        self.probabilities = (probabilities.detach() / sums.detach()).to(next(self.parameters()).device)
        try:
            yield
        finally:
            self.probabilities = previous

    def _mix(self, outputs: list[torch.Tensor]) -> torch.Tensor:
        probabilities = self.probabilities
        if probabilities is None or probabilities.shape[0] != len(outputs[0]):
            raise ValueError("Trunk mixture requires aligned image probabilities.")
        return (torch.stack(outputs, dim=1) * probabilities[:, :, None, None]).sum(dim=1)

    def _latents(
        self, views: torch.Tensor, mask_ratio: float, mask_noise: torch.Tensor | None
    ) -> tuple[list[torch.Tensor], torch.Tensor | None, torch.Tensor | None]:
        if mask_ratio > 0 and mask_noise is None:
            mask_noise = torch.rand(len(views), self.encoder.num_patches, device=views.device)
        first = self.encoder.forward_with_aux(views, mask_ratio, mask_noise=mask_noise)
        if self.shared_prefix:
            latents = [branch(first.tokens) for branch in self.branches]
        else:
            latents = [
                first.tokens,
                *(branch.forward_with_aux(views, mask_ratio, mask_noise=mask_noise).tokens for branch in self.branches),
            ]
        return latents, first.mask, first.ids_restore

    def reconstruct(self, views: torch.Tensor, mask_noise: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct using a single decoder or a probability-weighted prediction mixture."""
        latents, mask, ids_restore = self._latents(views, self.mask_ratio, mask_noise)
        if mask is None or ids_restore is None:
            raise ValueError("Trunk MAE reconstruction requires a positive mask ratio.")
        if self.additional_decoders:
            prediction = self._mix(
                [
                    decoder(latent, ids_restore)
                    for decoder, latent in zip([self.decoder, *self.additional_decoders], latents, strict=True)
                ]
            )
        else:
            prediction = self.decoder(self._mix(latents), ids_restore)
        return prediction, mask

    def _masked_recon_loss(
        self, imgs: torch.Tensor, *, augmented: bool = False, mask_noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        views = imgs if augmented else self.augment(imgs)
        prediction, mask = self.reconstruct(views, mask_noise)
        errors = (prediction - self._reconstruction_target(views)).square().mean(dim=-1)
        return errors, mask, views.new_zeros(())

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """Pool the GMM-weighted final encoder tokens for representation readouts."""
        with torch.no_grad():
            latents, _, _ = self._latents(imgs.to(next(self.parameters()).device), 0.0, None)
            return self._mix(latents)[:, 1:].mean(dim=1)
