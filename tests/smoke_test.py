"""Smoke test: build both MAE variants, forward, reconstruct, FLOPs, edges.

Run from project root:
    venv/Scripts/python.exe tests/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.mae import MaskedAutoencoder, build_mae  # noqa: E402
from utils.edge_tracker import (  # noqa: E402
    extract_edge_stats,
    sample_kan_edges,
    save_edge_snapshot,
)
from utils.metrics import estimate_flops  # noqa: E402


def make_cfg(decoder_type: str, in_chans: int = 1) -> dict:
    cfg = {
        "encoder": {
            "in_chans": in_chans,
            "patch_size": 16,
            "embed_dim": 384,
            "depth": 2,
            "num_heads": 6,
        },
        "decoder": {
            "type": decoder_type,
            "decoder_embed_dim": 256,
            "decoder_num_heads": 4,
        },
    }
    if decoder_type == "transformer":
        cfg["decoder"]["decoder_depth"] = 1
    else:
        cfg["decoder"].update(
            kan_hidden_dim=128, kan_grid_size=5, kan_spline_order=3,
        )
    return cfg


def test_forward_and_recon(name: str, mae: MaskedAutoencoder, x: torch.Tensor) -> None:
    out = mae(x)
    assert "loss" in out and torch.isfinite(out["loss"]), f"{name}: bad loss"
    assert out["pred"].dim() == 3, f"{name}: pred must be (B, N, P)"
    recon, mask, grid = mae.reconstruct(x)
    assert recon.shape == x.shape, f"{name}: recon shape mismatch"
    assert 0.7 < mask.float().mean().item() < 0.8, f"{name}: mask ratio off"
    print(f"  {name}: loss={out['loss'].item():.4f} recon={tuple(recon.shape)} grid={grid}")


def test_flops(name: str, mae: MaskedAutoencoder, x: torch.Tensor) -> int:
    flops = estimate_flops(mae, x[:1])
    assert flops is not None, f"{name}: estimate_flops returned None"
    assert isinstance(flops, int) and flops > 0, f"{name}: bad flop count {flops}"
    print(f"  {name}: estimated FLOPs (single sample) = {flops:,} ({flops / 1e9:.3f} GFLOPs)")
    return flops


def test_edges(name: str, mae: MaskedAutoencoder) -> None:
    stats = extract_edge_stats(mae.decoder, num_points=32)
    if name == "transformer":
        assert stats == {}, f"{name}: transformer should yield no KAN stats"
        print(f"  {name}: no KAN edges (expected)")
        return
    assert stats["n_layers"] == 2 and stats["n_edges"] > 0, f"{name}: bad stats {stats}"
    assert stats["edge_max_tv"] >= stats["edge_mean_tv"] >= 0.0
    edges = mae.decoder.get_edge_functions(num_points=32)
    assert set(edges.keys()) == {"kan1", "kan2"}, f"{name}: unexpected edge keys"
    bare = sample_kan_edges(mae.decoder.kan1, num_points=32)
    assert bare.shape == edges["kan1"].shape, f"{name}: bare sampler disagrees"
    print(
        f"  {name}: n_edges={stats['n_edges']} "
        f"mean_tv={stats['edge_mean_tv']:.4f} "
        f"max_tv={stats['edge_max_tv']:.4f}"
    )


def test_snapshot(mae: MaskedAutoencoder, tmp_dir: Path) -> None:
    snap_path = tmp_dir / "edges.npz"
    wrote = save_edge_snapshot(mae.decoder, snap_path)
    assert wrote, "snapshot did not write"
    assert snap_path.exists() and snap_path.stat().st_size > 0
    print(f"  snapshot saved: {snap_path.name} ({snap_path.stat().st_size} bytes)")


def main() -> None:
    torch.manual_seed(0)
    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")

    x = torch.randn(2, 1, 128, 128)

    print("\n--- build + forward + recon ---")
    cfg_t = make_cfg("transformer")
    cfg_k = make_cfg("kan")
    mae_t = build_mae(cfg_t)
    mae_k = build_mae(cfg_k)
    test_forward_and_recon("transformer", mae_t, x)
    test_forward_and_recon("kan", mae_k, x)

    print("\n--- estimate_flops ---")
    flops_t = test_flops("transformer", mae_t, x)
    flops_k = test_flops("kan", mae_k, x)
    assert flops_t > 1e6 and flops_k > 1e6, "flops suspiciously low"

    print("\n--- KAN edge tracking ---")
    test_edges("transformer", mae_t)
    test_edges("kan", mae_k)

    print("\n--- edge snapshot (KAN only) ---")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        test_snapshot(mae_k, Path(tmp))

    print("\n--- ECG (12-channel) sanity ---")
    x_ecg = torch.randn(2, 12, 128, 128)
    mae_k_ecg = build_mae(make_cfg("kan", in_chans=12))
    test_forward_and_recon("kan-ecg", mae_k_ecg, x_ecg)
    flops_ecg = test_flops("kan-ecg", mae_k_ecg, x_ecg)
    assert flops_ecg > flops_k, "12-ch should cost more than 1-ch"

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
