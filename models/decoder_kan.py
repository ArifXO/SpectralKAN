from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from efficient_kan import KANLinear


class KANDecoder(nn.Module):
    """KAN-based MAE decoder (project contribution).

    Same input/output contract as :class:`TransformerDecoder` so the two are
    swappable inside the MAE wrapper:

    * Input ``x``: ``(B, 1 + len_keep, encoder_embed_dim)`` from the encoder
      (cls token + visible-patch latents) and ``ids_restore``: ``(B, N)``.
    * Output: ``(B, N, patch_size * patch_size * in_chans)`` patch predictions
      in original spatial order.

    Architecture (hybrid):
      1. Linear projection from encoder dim to decoder dim.
      2. Mask tokens spliced into masked positions; gather restores order.
      3. **One** standard multi-head self-attention layer (pre-norm + residual)
         to mix information across patch positions.
      4. **Two** KAN layers (per-token) projecting
         ``decoder_embed_dim -> kan_hidden_dim -> patch_size**2 * in_chans``.
         The KAN edges are where interpretability lives.
    """

    def __init__(
        self,
        encoder_embed_dim: int = 384,
        decoder_embed_dim: int = 512,
        decoder_num_heads: int = 8,
        kan_hidden_dim: int = 512,
        kan_grid_size: int = 5,
        kan_spline_order: int = 3,
        patch_size: int = 16,
        in_chans: int = 1,
        max_patches: int = 1024,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.decoder_embed_dim = decoder_embed_dim
        self.kan_hidden_dim = kan_hidden_dim
        self.kan_grid_size = kan_grid_size
        self.kan_spline_order = kan_spline_order
        self.max_patches = max_patches

        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, max_patches + 1, decoder_embed_dim)
        )

        self.attn_norm = nn.LayerNorm(decoder_embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=decoder_embed_dim,
            num_heads=decoder_num_heads,
            batch_first=True,
        )

        self.kan_norm = nn.LayerNorm(decoder_embed_dim)
        out_dim = patch_size * patch_size * in_chans
        self.kan1 = KANLinear(
            decoder_embed_dim,
            kan_hidden_dim,
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
        )
        self.kan2 = KANLinear(
            kan_hidden_dim,
            out_dim,
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        for m in (self.decoder_embed, self.attn_norm, self.kan_norm):
            for p in m.modules():
                if isinstance(p, nn.Linear):
                    nn.init.xavier_uniform_(p.weight)
                    if p.bias is not None:
                        nn.init.zeros_(p.bias)
                elif isinstance(p, nn.LayerNorm):
                    nn.init.ones_(p.weight)
                    nn.init.zeros_(p.bias)
        # MultiheadAttention has its own xavier init by default; KANLinear
        # handles its own init in reset_parameters() — leave both alone.

    def forward(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(x)

        B, _, D = x.shape
        N = ids_restore.size(1)
        len_keep = x.size(1) - 1
        num_masked = N - len_keep

        if N + 1 > self.decoder_pos_embed.size(1):
            raise ValueError(
                f"Input requires {N} patch positions but decoder_pos_embed "
                f"only holds {self.decoder_pos_embed.size(1) - 1}. "
                "Increase max_patches."
            )

        mask_tokens = self.mask_token.expand(B, num_masked, D)
        x_no_cls = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_no_cls = torch.gather(
            x_no_cls, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D)
        )
        x = torch.cat([x[:, :1, :], x_no_cls], dim=1)

        x = x + self.decoder_pos_embed[:, : N + 1]

        h = self.attn_norm(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        x = self.kan_norm(x)
        x = self.kan1(x)
        x = self.kan2(x)

        return x[:, 1:, :]

    @torch.no_grad()
    def get_edge_functions(self, num_points: int = 200) -> dict[str, np.ndarray]:
        """Sample each KAN edge spline so it can be plotted/analyzed.

        For every ``KANLinear`` layer this returns the full per-edge response
        ``phi_{i,j}(x) = base_weight[j,i] * silu(x) + sum_k spline_weight[j,i,k] * B_k(x)``
        evaluated on a uniform grid that spans the layer's active spline range.

        Returns:
            ``{layer_name: array of shape (in_dim, out_dim, num_points)}``
            with one entry per KAN layer (``"kan1"``, ``"kan2"``).
        """
        edges: dict[str, np.ndarray] = {}
        for name, layer in [("kan1", self.kan1), ("kan2", self.kan2)]:
            edges[name] = self._extract_edges(layer, num_points)
        return edges

    @staticmethod
    def _extract_edges(layer: KANLinear, num_points: int) -> np.ndarray:
        in_f = layer.in_features
        out_f = layer.out_features
        order = layer.spline_order
        device = layer.grid.device
        dtype = layer.base_weight.dtype

        # Active range = where the spline basis is fully supported,
        # i.e. between the first and last "real" knots.
        grid_min = layer.grid[:, order].min().item()
        grid_max = layer.grid[:, -(order + 1)].max().item()
        xs = torch.linspace(grid_min, grid_max, num_points, device=device, dtype=dtype)

        # Broadcast xs across every input feature so b_splines evaluates each
        # feature's own basis at the same query points.
        x_input = xs.unsqueeze(1).expand(num_points, in_f).contiguous()
        basis = layer.b_splines(x_input)  # (T, in, num_basis)

        scaled = layer.scaled_spline_weight  # (out, in, num_basis)
        spline_term = torch.einsum("tik,jik->ijt", basis, scaled)

        base_act = layer.base_activation(xs)  # (T,)
        base_term = layer.base_weight.unsqueeze(-1) * base_act  # (out, in, T)
        base_term = base_term.permute(1, 0, 2)  # (in, out, T)

        return (spline_term + base_term).detach().cpu().numpy()

    def count_parameters(self) -> int:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"KANDecoder trainable parameters: {total:,}")
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

    decoder = KANDecoder(
        encoder_embed_dim=encoder_embed_dim,
        decoder_embed_dim=512,
        decoder_num_heads=8,
        kan_hidden_dim=512,
        kan_grid_size=5,
        kan_spline_order=3,
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

    edges = decoder.get_edge_functions(num_points=200)
    for name, arr in edges.items():
        print(f"edges[{name!r}]: shape={arr.shape}, dtype={arr.dtype}")

    decoder.count_parameters()
