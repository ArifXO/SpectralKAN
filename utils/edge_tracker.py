"""KAN edge analysis helpers.

These functions read a decoder's KAN spline edges and either summarise them
(``extract_edge_stats``) or freeze them to disk for later visualisation
(``save_edge_snapshot``). Both are no-ops on non-KAN decoders so the same
callsite works for both baselines and our contribution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

try:
    from efficient_kan import KANLinear
except ImportError:  # keep importable in environments without efficient-kan
    KANLinear = None  # type: ignore[assignment, misc]


def _find_kan_layers(module: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return ``[(qualified_name, layer)]`` for every KANLinear in ``module``."""
    if KANLinear is None:
        return []
    return [
        (name, sub)
        for name, sub in module.named_modules()
        if isinstance(sub, KANLinear)
    ]


def extract_edge_stats(decoder: nn.Module, num_points: int = 200) -> dict[str, Any]:
    """Mean and max total variation across every spline edge in ``decoder``.

    Total variation of one edge ``f`` sampled on a grid ``x_0..x_{T-1}`` is
    ``sum_i |f(x_{i+1}) - f(x_i)|`` — a curvature-free measure of how much
    the spline "moves" over its active range. Larger TV means a more
    expressive edge; a flat near-zero edge has TV ~ 0 and is prunable.

    Returns ``{}`` if the decoder has no KAN layers, so callers can blindly
    call this on the transformer baseline.

    Output keys: ``edge_mean_tv``, ``edge_max_tv``, ``n_edges``, ``n_layers``.
    """
    kan_layers = _find_kan_layers(decoder)
    if not kan_layers:
        return {}

    if hasattr(decoder, "get_edge_functions"):
        edges_by_layer = decoder.get_edge_functions(num_points=num_points)
        arrays = list(edges_by_layer.values())
    else:
        arrays = [
            sample_kan_edges(layer, num_points) for _name, layer in kan_layers
        ]

    tv_values: list[np.ndarray] = []
    for arr in arrays:
        # arr shape: (in, out, T) — TV along T, flatten to one value per edge.
        diffs = np.abs(np.diff(arr, axis=-1))
        tv = diffs.sum(axis=-1).reshape(-1)
        tv_values.append(tv)

    all_tv = np.concatenate(tv_values) if tv_values else np.zeros(0)
    return {
        "edge_mean_tv": float(all_tv.mean()) if all_tv.size else 0.0,
        "edge_max_tv": float(all_tv.max()) if all_tv.size else 0.0,
        "n_edges": int(all_tv.size),
        "n_layers": len(arrays),
    }


@torch.no_grad()
def sample_kan_edges(layer: nn.Module, num_points: int = 200) -> np.ndarray:
    """Evaluate every edge of a single ``KANLinear`` on a uniform grid.

    For ``phi_{i,j}(x) = base_weight[j,i] * silu(x) +
    sum_k spline_weight[j,i,k] * B_k(x)``, returns an
    ``(in_features, out_features, num_points)`` array sampled across the
    layer's active spline range. Used by ``KANDecoder.get_edge_functions``
    and as the fallback path inside :func:`extract_edge_stats`.
    """
    in_f = layer.in_features
    order = layer.spline_order
    device = layer.grid.device
    dtype = layer.base_weight.dtype

    grid_min = layer.grid[:, order].min().item()
    grid_max = layer.grid[:, -(order + 1)].max().item()
    xs = torch.linspace(grid_min, grid_max, num_points, device=device, dtype=dtype)

    x_input = xs.unsqueeze(1).expand(num_points, in_f).contiguous()
    basis = layer.b_splines(x_input)
    scaled = layer.scaled_spline_weight
    spline_term = torch.einsum("tik,jik->ijt", basis, scaled)

    base_act = layer.base_activation(xs)
    base_term = (layer.base_weight.unsqueeze(-1) * base_act).permute(1, 0, 2)

    return (spline_term + base_term).detach().cpu().numpy()


@torch.no_grad()
def save_edge_snapshot(decoder: nn.Module, path: str | Path) -> bool:
    """Freeze every KAN layer's raw spline parameters to ``path`` as ``.npz``.

    Saves, per layer ``L``: ``L__grid``, ``L__base_weight``, ``L__spline_weight``,
    ``L__spline_scaler`` (if present), plus scalar metadata
    (``L__in_features``, ``L__out_features``, ``L__spline_order``,
    ``L__grid_size``). Layer names are sanitised (``.`` → ``__``).

    Returns ``True`` if anything was written, ``False`` for non-KAN decoders.
    Loaders can later replay the spline by combining ``base_weight`` and
    ``spline_weight * spline_scaler`` over the saved grid.
    """
    kan_layers = _find_kan_layers(decoder)
    if not kan_layers:
        return False

    arrays: dict[str, np.ndarray] = {}
    for raw_name, layer in kan_layers:
        key = (raw_name or "kan").replace(".", "__")

        arrays[f"{key}__grid"] = layer.grid.detach().cpu().numpy()
        arrays[f"{key}__base_weight"] = layer.base_weight.detach().cpu().numpy()
        arrays[f"{key}__spline_weight"] = layer.spline_weight.detach().cpu().numpy()
        scaler = getattr(layer, "spline_scaler", None)
        if isinstance(scaler, torch.Tensor):
            arrays[f"{key}__spline_scaler"] = scaler.detach().cpu().numpy()

        arrays[f"{key}__in_features"] = np.int64(layer.in_features)
        arrays[f"{key}__out_features"] = np.int64(layer.out_features)
        arrays[f"{key}__spline_order"] = np.int64(layer.spline_order)
        grid_size = layer.grid.shape[1] - 2 * layer.spline_order - 1
        arrays[f"{key}__grid_size"] = np.int64(grid_size)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)
    return True


if __name__ == "__main__":
    import sys
    import tempfile

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    torch.manual_seed(0)

    print("--- KAN decoder ---")
    from models.decoder_kan import KANDecoder

    decoder = KANDecoder(
        encoder_embed_dim=64,
        decoder_embed_dim=32,
        decoder_num_heads=4,
        kan_hidden_dim=24,
        kan_grid_size=5,
        kan_spline_order=3,
        patch_size=8,
        in_chans=1,
        max_patches=64,
    )

    stats = extract_edge_stats(decoder, num_points=64)
    print(f"  stats: {stats}")
    expected_n_edges = (32 * 24) + (24 * 8 * 8 * 1)
    assert stats["n_layers"] == 2, stats
    assert stats["n_edges"] == expected_n_edges, (stats["n_edges"], expected_n_edges)
    assert stats["edge_mean_tv"] >= 0.0
    assert stats["edge_max_tv"] >= stats["edge_mean_tv"]

    with tempfile.TemporaryDirectory() as tmp:
        snap_path = Path(tmp) / "edges.npz"
        wrote = save_edge_snapshot(decoder, snap_path)
        assert wrote is True
        assert snap_path.exists() and snap_path.stat().st_size > 0
        with np.load(snap_path) as loaded:
            keys = set(loaded.files)
            for layer in ("kan1", "kan2"):
                for suffix in (
                    "grid",
                    "base_weight",
                    "spline_weight",
                    "in_features",
                    "out_features",
                    "spline_order",
                    "grid_size",
                ):
                    assert f"{layer}__{suffix}" in keys, f"missing {layer}__{suffix}"
        print(f"  snapshot keys: {sorted(keys)}")
        print(f"  snapshot bytes: {snap_path.stat().st_size}")

    print("--- non-KAN decoder ---")
    from models.decoder_transformer import TransformerDecoder

    t_decoder = TransformerDecoder(
        encoder_embed_dim=64,
        decoder_embed_dim=32,
        decoder_depth=1,
        decoder_num_heads=4,
        patch_size=8,
        in_chans=1,
        max_patches=64,
    )
    t_stats = extract_edge_stats(t_decoder)
    assert t_stats == {}, t_stats
    print(f"  stats (transformer): {t_stats}")

    with tempfile.TemporaryDirectory() as tmp:
        snap_path = Path(tmp) / "edges_transformer.npz"
        wrote = save_edge_snapshot(t_decoder, snap_path)
        assert wrote is False
        assert not snap_path.exists()
        print(f"  snapshot wrote={wrote} (file not created, as expected)")

    print("--- bare nn.Module ---")
    bare = nn.Linear(8, 4)
    assert extract_edge_stats(bare) == {}
    assert save_edge_snapshot(bare, Path(tempfile.gettempdir()) / "noop.npz") is False
    print("  ok")

    print("\nAll tests passed.")
