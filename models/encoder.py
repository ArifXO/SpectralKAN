from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block


class SpectralViTEncoder(nn.Module):
    """ViT-Small encoder with random patch masking for MAE pretraining.

    Accepts spectrograms of shape (B, C, H, W) with arbitrary spatial sizes.
    Inputs are zero-padded so H and W are multiples of ``patch_size``.
    """

    def __init__(
        self,
        in_chans: int = 1,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        masking_ratio: float = 0.75,
        max_patches: int = 1024,
    ) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.masking_ratio = masking_ratio
        self.max_patches = max_patches

        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches + 1, embed_dim))

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view(w.size(0), -1))
        nn.init.zeros_(self.patch_embed.bias)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @contextmanager
    def masking_disabled(self):
        """Temporarily set ``masking_ratio=0`` so the encoder returns all patches."""
        saved = self.masking_ratio
        self.masking_ratio = 0.0
        try:
            yield
        finally:
            self.masking_ratio = saved

    def patchify(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        ps = self.patch_size
        _, _, H, W = x.shape
        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        tokens = self.patch_embed(x)
        h, w = tokens.shape[-2], tokens.shape[-1]
        tokens = tokens.flatten(2).transpose(1, 2)
        return tokens, (h, w)

    def random_masking(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        len_keep = max(1, int(round(N * (1.0 - self.masking_ratio))))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_kept = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0.0
        mask = torch.gather(mask, 1, ids_restore)
        return x_kept, mask, ids_restore

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
        tokens, grid = self.patchify(x)
        N = tokens.size(1)
        if N + 1 > self.pos_embed.size(1):
            raise ValueError(
                f"Input produces {N} patches but pos_embed only holds {self.pos_embed.size(1) - 1}. "
                f"Increase max_patches."
            )

        tokens = tokens + self.pos_embed[:, 1 : N + 1]
        tokens, mask, ids_restore = self.random_masking(tokens)

        cls = self.cls_token + self.pos_embed[:, :1]
        cls = cls.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        return tokens, mask, ids_restore, grid

    def patch_to_img(
        self, patches: torch.Tensor, grid_size: tuple[int, int]
    ) -> torch.Tensor:
        """Fold (B, N, patch_size*patch_size*C) patch predictions back to (B, C, H, W)."""
        h, w = grid_size
        B, N, _ = patches.shape
        if N != h * w:
            raise ValueError(f"Got {N} patches but grid is {h}x{w}={h * w}.")
        ps = self.patch_size
        C = self.in_chans
        x = patches.reshape(B, h, w, ps, ps, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.reshape(B, C, h * ps, w * ps)


if __name__ == "__main__":
    torch.manual_seed(0)
    encoder = SpectralViTEncoder(in_chans=1, patch_size=16)
    fake = torch.randn(4, 1, 128, 128)
    latent, mask, ids_restore, grid = encoder(fake)
    print(f"input shape:       {tuple(fake.shape)}")
    print(f"patch grid:        {grid} -> {grid[0] * grid[1]} patches")
    print(f"latent shape:      {tuple(latent.shape)} (cls + visible)")
    print(f"mask shape:        {tuple(mask.shape)} (1=masked, 0=kept)")
    print(f"ids_restore shape: {tuple(ids_restore.shape)}")
    print(f"masked fraction:   {mask.float().mean().item():.3f}")
