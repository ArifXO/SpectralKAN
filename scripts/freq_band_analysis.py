"""Per-band reconstruction MSE for the Claim 1 evaluation.

Runs the test set through two MAE checkpoints (transformer-decoder and
KAN-decoder), composes the reconstructions via ``MaskedAutoencoder.reconstruct``,
and reports MSE in three frequency bands so we can ask whether the KAN
decoder distributes error differently across the spectrum.

Two band partitions are reported:

* **Equal thirds** along the spectrogram frequency axis (mel for ESC-50,
  STFT bins for PTB-XL). Indices come from ``F // 3`` and ``2F // 3``,
  matching ``MaskedAutoencoder.compute_frequency_band_loss``.
* **Clinical bands** (PTB-XL only): 0-5 Hz, 5-15 Hz, 15-40 Hz, mapped to
  STFT bin ranges using ``sample_rate`` and ``n_fft``. Equivalent in this
  setting to a scipy ``butter`` bandpass: PTB-XL stores ``stft.abs()`` so
  phase is gone and the only faithful "filter" is bin slicing on magnitude.

Outputs (under ``--output-dir``, or under the checkpoint run folder by default):
    freq_band_mse_<dataset>_equal.png
    freq_band_mse_<dataset>_clinical.png   # ptbxl only
    freq_band_mse_<dataset>.csv

Usage:
    python scripts/freq_band_analysis.py \
        --transformer-checkpoint results/<run_t>/checkpoints/last.pt \
        --kan-checkpoint         results/<run_k>/checkpoints/last.pt \
        --dataset ptbxl
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.mae import build_mae  # noqa: E402
from output_paths import infer_common_run_dir  # noqa: E402
from train import bridge_config  # noqa: E402  -- shared YAML→build_mae translator


def load_test_loader(
    dataset: str, batch_size: int, num_workers: int, data_root: str
):
    if dataset == "esc50":
        from data.audio_dataset import get_esc50_loaders

        _train, _val, test = get_esc50_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
            root=data_root,
            val_fold=None,
        )
        return test, 1

    if dataset == "ptbxl":
        from data.ecg_dataset import get_ptbxl_loaders

        _train, _val, test = get_ptbxl_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
            root=data_root,
            download=False,
        )
        return test, 12

    raise ValueError(f"Unknown dataset {dataset!r}")


def load_mae(ckpt_path: str, in_chans: int, device: torch.device) -> torch.nn.Module:
    print(f"loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    raw_cfg = ckpt.get("config", {})
    if not raw_cfg:
        print(f"[warn] {ckpt_path} has no 'config'; using default model dims.")
    bridged = bridge_config(raw_cfg, in_chans=in_chans)
    model = build_mae(bridged)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(
            f"[warn] state_dict load: missing={len(missing)} unexpected={len(unexpected)}"
        )
    return model.to(device).eval()


def equal_thirds(f_size: int) -> dict[str, tuple[int, int]]:
    b1 = f_size // 3
    b2 = 2 * f_size // 3
    return {"low": (0, b1), "mid": (b1, b2), "high": (b2, f_size)}


def hz_band_slices(
    sample_rate: int, n_fft: int, n_bins: int, edges_hz: list[float]
) -> list[tuple[int, int]]:
    """Convert ``[hz0, hz1, ..., hzK]`` into K bin ranges, clipped to ``n_bins``."""
    bins = [round(hz * n_fft / sample_rate) for hz in edges_hz]
    bins = [min(max(b, 0), n_bins) for b in bins]
    return [(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]


@torch.no_grad()
def joint_band_mse(
    models: dict[str, torch.nn.Module],
    loader,
    device: torch.device,
    use_amp: bool,
    band_groups: dict[str, dict[str, tuple[int, int]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Single test-set pass; per-band SSE + pixel counts for each model.

    Returns ``out[model][group][band] = mse``.
    """
    sse: dict = {m: {g: {b: 0.0 for b in bands} for g, bands in band_groups.items()} for m in models}
    px: dict = {m: {g: {b: 0 for b in bands} for g, bands in band_groups.items()} for m in models}

    for x, _y in tqdm(loader, desc="recon", dynamic_ncols=True):
        x = x.to(device, non_blocking=True)
        x_ref = x.float()
        for m_name, model in models.items():
            with torch.amp.autocast("cuda", enabled=use_amp):
                recon, _mask, _grid = model.reconstruct(x)
            diff2 = (recon.float() - x_ref) ** 2  # (B, C, F, T) — F on dim=-2
            for g_name, bands in band_groups.items():
                for b_name, (lo, hi) in bands.items():
                    chunk = diff2[..., lo:hi, :]
                    sse[m_name][g_name][b_name] += chunk.sum().item()
                    px[m_name][g_name][b_name] += chunk.numel()

    return {
        m: {
            g: {
                b: sse[m][g][b] / max(px[m][g][b], 1)
                for b in bands
            }
            for g, bands in band_groups.items()
        }
        for m in models
    }


def plot_bands(
    transformer_vals: dict[str, float],
    kan_vals: dict[str, float],
    title: str,
    output_path: Path,
) -> None:
    bands = list(transformer_vals.keys())
    x = np.arange(len(bands))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(6.0, 1.4 * len(bands) + 2.5), 4.5))
    ax.bar(x - width / 2, [transformer_vals[b] for b in bands], width, label="Transformer decoder")
    ax.bar(x + width / 2, [kan_vals[b] for b in bands], width, label="KAN decoder")
    ax.set_xticks(x, bands)
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-band reconstruction MSE for Claim 1")
    p.add_argument("--transformer-checkpoint", required=True, help="Path to transformer-decoder MAE .pt")
    p.add_argument("--kan-checkpoint", required=True, help="Path to KAN-decoder MAE .pt")
    p.add_argument("--dataset", required=True, choices=["esc50", "ptbxl"])
    p.add_argument("--data-root", default=None, help="Override default data root")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Defaults to <checkpoint-run>/freq_band when both "
            "checkpoints share a run, otherwise results/freq_band/<timestamp>_<dataset>."
        ),
    )
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    data_root = args.data_root or (
        "data/ESC-50" if args.dataset == "esc50" else "data/ptbxl"
    )
    test_loader, in_chans = load_test_loader(
        args.dataset, args.batch_size, args.num_workers, data_root
    )

    sample_x, _ = next(iter(test_loader))
    f_size = sample_x.size(-2)
    print(f"dataset={args.dataset} | freq-axis size={f_size} | channels={in_chans}")

    band_groups: dict[str, dict[str, tuple[int, int]]] = {
        "equal_thirds": equal_thirds(f_size),
    }

    if args.dataset == "ptbxl":
        from data.ecg_dataset import PTBXLDataset

        sr = PTBXLDataset.SAMPLE_RATE
        n_fft = PTBXLDataset.N_FFT
        edges_hz = [0.0, 5.0, 15.0, 40.0]
        spans = hz_band_slices(sr, n_fft, f_size, edges_hz)
        band_groups["clinical"] = {
            "0-5Hz": spans[0],
            "5-15Hz": spans[1],
            "15-40Hz": spans[2],
        }
        print(f"clinical bin ranges (sr={sr}, n_fft={n_fft}): {band_groups['clinical']}")

    use_amp = device.type == "cuda" and not args.no_amp
    print(f"device={device} | AMP={'on' if use_amp else 'off'}")

    transformer_checkpoint = Path(args.transformer_checkpoint)
    kan_checkpoint = Path(args.kan_checkpoint)

    mae_t = load_mae(str(transformer_checkpoint), in_chans, device)
    mae_k = load_mae(str(kan_checkpoint), in_chans, device)

    print("running test-set reconstruction (both models)...")
    results = joint_band_mse(
        {"transformer": mae_t, "kan": mae_k},
        test_loader, device, use_amp, band_groups,
    )

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        run_dir = infer_common_run_dir([transformer_checkpoint, kan_checkpoint])
        if run_dir is not None:
            output_dir = run_dir / "freq_band"
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("results") / "freq_band" / f"{stamp}_{args.dataset}"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"freq_band_mse_{args.dataset}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "band_type", "band", "decoder", "mse"])
        for g_name, bands in band_groups.items():
            for b_name in bands:
                for m_name in ("transformer", "kan"):
                    writer.writerow([
                        args.dataset, g_name, b_name, m_name,
                        f"{results[m_name][g_name][b_name]:.6f}",
                    ])

    eq_png = output_dir / f"freq_band_mse_{args.dataset}_equal.png"
    plot_bands(
        results["transformer"]["equal_thirds"],
        results["kan"]["equal_thirds"],
        f"Per-band reconstruction MSE — {args.dataset} (equal thirds)",
        eq_png,
    )
    print(f"  -> {eq_png}")

    if "clinical" in band_groups:
        cl_png = output_dir / f"freq_band_mse_{args.dataset}_clinical.png"
        plot_bands(
            results["transformer"]["clinical"],
            results["kan"]["clinical"],
            f"Per-band reconstruction MSE — {args.dataset} (clinical bands)",
            cl_png,
        )
        print(f"  -> {cl_png}")

    print(f"  -> {csv_path}")
    for g_name, bands in band_groups.items():
        print(f"\n[{g_name}]")
        header = f"  {'band':<10} {'transformer':>14} {'kan':>14}  delta(kan-tr)"
        print(header)
        for b_name in bands:
            t_v = results["transformer"][g_name][b_name]
            k_v = results["kan"][g_name][b_name]
            print(f"  {b_name:<10} {t_v:14.6f} {k_v:14.6f} {(k_v - t_v):+10.6f}")


if __name__ == "__main__":
    main()
