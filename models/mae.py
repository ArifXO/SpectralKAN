from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import compute_band_mse_tensors

try:
    from .encoder import SpectralViTEncoder
    from .decoder_transformer import TransformerDecoder
    from .decoder_kan import KANDecoder
except ImportError:
    from encoder import SpectralViTEncoder
    from decoder_transformer import TransformerDecoder
    from decoder_kan import KANDecoder


class MaskedAutoencoder(nn.Module):
    """MAE wrapper: encoder + decoder + reconstruction loss on masked patches.

    The encoder picks a fresh random 75% mask each forward; the decoder
    predicts every patch but the loss only sees masked positions, matching
    He et al. 2022. ``norm_pix_loss`` switches the target to per-patch
    z-scored pixels (the original-paper recipe — small accuracy bump).

    Encoder/decoder are constructor args so the same wrapper covers both
    the transformer baseline and the KAN decoder we are evaluating.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        norm_pix_loss: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.norm_pix_loss = norm_pix_loss
        self.patch_size = encoder.patch_size
        self.in_chans = encoder.in_chans

    def patchify_target(self, x: torch.Tensor) -> torch.Tensor:
        """Pad ``(B, C, H, W)`` and split into ``(B, N, ps*ps*C)`` raw patches.

        The pad-to-multiple-of-``patch_size`` step mirrors the encoder, and
        the per-patch flatten order is the inverse of ``encoder.patch_to_img``,
        so target patches line up with what the decoder predicts.
        """
        ps = self.patch_size
        _, _, H, W = x.shape
        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        B, C, H, W = x.shape
        h, w = H // ps, W // ps
        x = x.reshape(B, C, h, ps, w, ps).permute(0, 2, 4, 3, 5, 1).contiguous()
        return x.reshape(B, h * w, ps * ps * C)

    def _normalize_target(self, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True, unbiased=False)
        std = (var + 1e-6).sqrt()
        return (target - mean) / std, mean, std

    def forward(self, x: torch.Tensor, return_reconstruction: bool = False) -> dict:
        target_raw = self.patchify_target(x)
        target = target_raw

        latent, mask, ids_restore, grid = self.encoder(x)
        pred = self.decoder(latent, ids_restore)

        if self.norm_pix_loss:
            target, mean, std = self._normalize_target(target)
        else:
            mean = std = None

        loss_per_patch = (pred - target).pow(2).mean(dim=-1)
        loss = (loss_per_patch * mask).sum() / mask.sum().clamp(min=1.0)

        out = {
            "loss": loss,
            "pred": pred,
            "target": target,
            "mask": mask,
            "ids_restore": ids_restore,
            "grid": grid,
        }
        if return_reconstruction:
            pred_pixels = pred * std + mean if self.norm_pix_loss else pred
            out["recon"] = self._compose_reconstruction(
                target_raw,
                pred_pixels,
                mask,
                grid,
                original_size=x.shape[-2:],
            )
        return out

    def _compose_reconstruction(
        self,
        target_patches: torch.Tensor,
        pred_patches: torch.Tensor,
        mask: torch.Tensor,
        grid: tuple[int, int],
        original_size: tuple[int, int],
    ) -> torch.Tensor:
        keep = mask.unsqueeze(-1)
        composed = target_patches * (1.0 - keep) + pred_patches * keep
        recon = self.encoder.patch_to_img(composed, grid)
        h_orig, w_orig = original_size
        return recon[..., :h_orig, :w_orig]

    @torch.no_grad()
    def reconstruct(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        """Run encoder+decoder and fold predictions back to a full spectrogram.

        Visible patches keep their original pixels; masked patches are filled
        from the decoder. The output is cropped back to the input H/W in case
        the encoder padded internally.

        Returns ``(recon, mask, grid)``: ``recon`` is ``(B, C, H, W)``, ``mask``
        is ``(B, N)`` with 1 at masked positions, ``grid`` is the padded patch
        grid ``(h, w)``.
        """
        out = self.forward(x, return_reconstruction=True)
        return out["recon"], out["mask"], out["grid"]

    @staticmethod
    def compute_frequency_band_loss(
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        freq_dim: int = -2,
    ) -> dict[str, torch.Tensor]:
        """Per-band MSE for low / mid / high frequency thirds.

        Spectrograms here are ``(B, C, H, W)`` with frequency on H (the standard
        torchaudio / librosa convention), so the default splits along ``dim=-2``.
        For data laid out with frequency on a different axis, override ``freq_dim``.

        Used by the Claim 1 evaluation to check whether the KAN decoder
        distributes reconstruction error differently across frequency bands.
        """
        return compute_band_mse_tensors(
            reconstructed,
            original,
            n_bands=3,
            freq_dim=freq_dim,
            names=["low", "mid", "high"],
        )


def build_mae(config: dict) -> MaskedAutoencoder:
    """Build an ``MaskedAutoencoder`` from a config dict (e.g. parsed YAML).

    Expected config shape::

        encoder:
          in_chans, patch_size, embed_dim, depth, num_heads,
          mlp_ratio, masking_ratio, max_patches      # all optional
        decoder:
          type: "transformer" | "kan"
          # transformer-only: decoder_depth, mlp_ratio
          # kan-only:         kan_hidden_dim, kan_grid_size, kan_spline_order
          # shared:           decoder_embed_dim, decoder_num_heads, max_patches
        norm_pix_loss: bool                          # optional, default True

    The factory copies ``encoder_embed_dim``, ``patch_size``, and ``in_chans``
    from the encoder into the decoder kwargs (unless explicitly overridden),
    so the contract between the two halves stays consistent.
    """
    enc_cfg = dict(config.get("encoder", {}))
    dec_cfg = dict(config.get("decoder", {}))

    encoder = SpectralViTEncoder(**enc_cfg)

    dec_type = dec_cfg.pop("type", "transformer")
    dec_cfg.setdefault("encoder_embed_dim", encoder.embed_dim)
    dec_cfg.setdefault("patch_size", encoder.patch_size)
    dec_cfg.setdefault("in_chans", encoder.in_chans)

    if dec_type == "transformer":
        decoder: nn.Module = TransformerDecoder(**dec_cfg)
    elif dec_type == "kan":
        decoder = KANDecoder(**dec_cfg)
    else:
        raise ValueError(
            f"Unknown decoder type {dec_type!r} (expected 'transformer' or 'kan')"
        )

    return MaskedAutoencoder(
        encoder=encoder,
        decoder=decoder,
        norm_pix_loss=config.get("norm_pix_loss", True),
    )


if __name__ == "__main__":
    torch.manual_seed(0)

    fake = torch.randn(2, 1, 128, 128)

    cfg_t = {
        "encoder": {
            "in_chans": 1,
            "patch_size": 16,
            "embed_dim": 384,
            "depth": 2,
            "num_heads": 6,
        },
        "decoder": {
            "type": "transformer",
            "decoder_embed_dim": 256,
            "decoder_depth": 1,
            "decoder_num_heads": 4,
        },
    }
    mae_t = build_mae(cfg_t)
    out_t = mae_t(fake)
    recon_t, mask_t, grid_t = mae_t.reconstruct(fake)
    bands_t = MaskedAutoencoder.compute_frequency_band_loss(fake, recon_t)
    print("=== TransformerDecoder MAE ===")
    print(f"input:           {tuple(fake.shape)}")
    print(f"loss:            {out_t['loss'].item():.4f}")
    print(f"pred:            {tuple(out_t['pred'].shape)}")
    print(f"mask (mean):     {out_t['mask'].float().mean().item():.3f}")
    print(f"recon:           {tuple(recon_t.shape)} (grid={grid_t})")
    print(
        f"band MSE:        low={bands_t['low'].item():.4f} "
        f"mid={bands_t['mid'].item():.4f} high={bands_t['high'].item():.4f}"
    )
    total_t = sum(p.numel() for p in mae_t.parameters() if p.requires_grad)
    print(f"total params:    {total_t:,}")

    cfg_k = {
        "encoder": {
            "in_chans": 1,
            "patch_size": 16,
            "embed_dim": 384,
            "depth": 2,
            "num_heads": 6,
        },
        "decoder": {
            "type": "kan",
            "decoder_embed_dim": 256,
            "decoder_num_heads": 4,
            "kan_hidden_dim": 128,
            "kan_grid_size": 5,
            "kan_spline_order": 3,
        },
    }
    mae_k = build_mae(cfg_k)
    out_k = mae_k(fake)
    recon_k, mask_k, grid_k = mae_k.reconstruct(fake)
    bands_k = MaskedAutoencoder.compute_frequency_band_loss(fake, recon_k)
    print()
    print("=== KANDecoder MAE ===")
    print(f"input:           {tuple(fake.shape)}")
    print(f"loss:            {out_k['loss'].item():.4f}")
    print(f"pred:            {tuple(out_k['pred'].shape)}")
    print(f"mask (mean):     {out_k['mask'].float().mean().item():.3f}")
    print(f"recon:           {tuple(recon_k.shape)} (grid={grid_k})")
    print(
        f"band MSE:        low={bands_k['low'].item():.4f} "
        f"mid={bands_k['mid'].item():.4f} high={bands_k['high'].item():.4f}"
    )
    total_k = sum(p.numel() for p in mae_k.parameters() if p.requires_grad)
    print(f"total params:    {total_k:,}")

    assert out_t["pred"].shape == out_k["pred"].shape, "decoder outputs must match shape"
    assert recon_t.shape == fake.shape and recon_k.shape == fake.shape, "recon must match input shape"
    print("\nshape contract: transformer & kan MAE produce identical pred/recon shapes.")
