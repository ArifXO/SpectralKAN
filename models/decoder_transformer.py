from __future__ import annotations

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

try:
    from .decoder_utils import splice_mask_tokens
except ImportError:
    from decoder_utils import splice_mask_tokens


class TransformerDecoder(nn.Module):
    """Baseline MAE decoder.

    Takes the encoder's visible-token latents plus ``ids_restore``,
    re-inserts learned mask tokens at the masked positions, runs a small
    transformer stack, and predicts one flattened patch
    (``patch_size * patch_size * in_chans``) per token.
    """

    def __init__(
        self,
        encoder_embed_dim: int = 384,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 2,
        decoder_num_heads: int = 8,
        mlp_ratio: float = 4.0,
        patch_size: int = 16,
        in_chans: int = 1,
        max_patches: int = 1024,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.decoder_embed_dim = decoder_embed_dim
        self.max_patches = max_patches

        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, max_patches + 1, decoder_embed_dim)
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.head = nn.Linear(
            decoder_embed_dim, patch_size * patch_size * in_chans, bias=True
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """Reconstruct flattened patches for every position.

        Args:
            x: ``(B, 1 + len_keep, encoder_embed_dim)`` from the encoder
                (cls token followed by visible-patch latents).
            ids_restore: ``(B, N)`` permutation that returns shuffled tokens
                back to their original patch order.

        Returns:
            ``(B, N, patch_size * patch_size * in_chans)`` patch predictions
            in original spatial order. The MAE wrapper restricts the
            reconstruction loss to masked positions.
        """
        x = self.decoder_embed(x)
        x = splice_mask_tokens(x, ids_restore, self.mask_token, self.decoder_pos_embed)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        x = self.head(x)
        return x[:, 1:, :]

    def count_parameters(self) -> int:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"TransformerDecoder trainable parameters: {total:,}")
        return total


if __name__ == "__main__":
    torch.manual_seed(0)

    B = 4
    grid = (8, 8)
    N = grid[0] * grid[1]
    encoder_embed_dim = 384
    masking_ratio = 0.75
    patch_size = 16
    in_chans = 1
    len_keep = max(1, int(round(N * (1.0 - masking_ratio))))

    fake_latent = torch.randn(B, 1 + len_keep, encoder_embed_dim)
    fake_ids_restore = torch.stack([torch.randperm(N) for _ in range(B)])

    decoder = TransformerDecoder(
        encoder_embed_dim=encoder_embed_dim,
        decoder_embed_dim=512,
        decoder_depth=2,
        decoder_num_heads=8,
        mlp_ratio=4.0,
        patch_size=patch_size,
        in_chans=in_chans,
    )
    pred = decoder(fake_latent, fake_ids_restore)

    print(f"fake encoder latent: {tuple(fake_latent.shape)} (cls + visible)")
    print(f"fake ids_restore:    {tuple(fake_ids_restore.shape)}")
    print(f"decoder output:      {tuple(pred.shape)}")
    print(
        f"  -> (B, N, patch_size*patch_size*C) = "
        f"({B}, {N}, {patch_size * patch_size * in_chans})"
    )
    decoder.count_parameters()
