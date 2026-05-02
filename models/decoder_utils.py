"""Shared MAE decoder helpers.

The two decoders (transformer baseline, KAN ours) share the
same boilerplate for re-inserting mask tokens at the masked
positions and adding decoder positional embeddings. Centralised
here so the two stay in lockstep.
"""

from __future__ import annotations

import torch


def splice_mask_tokens(
    x: torch.Tensor,
    ids_restore: torch.Tensor,
    mask_token: torch.Tensor,
    decoder_pos_embed: torch.Tensor,
) -> torch.Tensor:
    """Re-insert mask tokens, restore patch order, add positional embedding.

    Args:
        x: ``(B, 1 + len_keep, D)`` cls token followed by visible-patch latents,
            already projected into the decoder dimension.
        ids_restore: ``(B, N)`` permutation that returns shuffled tokens
            back to their original patch order.
        mask_token: ``(1, 1, D)`` learned mask-token parameter.
        decoder_pos_embed: ``(1, max_patches + 1, D)`` learned positional table.

    Returns:
        ``(B, 1 + N, D)`` with cls token first, then patches in original order,
        with positional embeddings added.
    """
    B, _, D = x.shape
    N = ids_restore.size(1)
    len_keep = x.size(1) - 1
    num_masked = N - len_keep

    if N + 1 > decoder_pos_embed.size(1):
        raise ValueError(
            f"Input requires {N} patch positions but decoder_pos_embed "
            f"only holds {decoder_pos_embed.size(1) - 1}. Increase max_patches."
        )

    mask_tokens = mask_token.expand(B, num_masked, D)
    x_no_cls = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
    x_no_cls = torch.gather(
        x_no_cls, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D)
    )
    x = torch.cat([x[:, :1, :], x_no_cls], dim=1)
    return x + decoder_pos_embed[:, : N + 1]
