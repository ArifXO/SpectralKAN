"""MAE pretraining entry point.

Usage:
    python scripts/train.py --config configs/esc50_config.yaml
    python scripts/train.py --config configs/ptbxl_config.yaml --resume results/<run>/checkpoints/last.pt

The training loop matches the He et al. 2022 MAE recipe: AdamW (betas 0.9/0.95),
cosine schedule with linear warmup, mixed precision on CUDA, gradient clipping
at 1.0. Per-epoch metrics (total loss + low/mid/high-band reconstruction MSE)
are written to a fresh ``results/<timestamp>_<dataset>_<decoder>/`` run folder.
Checkpoints land in that folder's ``checkpoints/`` subdirectory every
``logging.save_every`` epochs and on the final epoch.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.mae import MaskedAutoencoder, build_mae  # noqa: E402
from output_paths import infer_run_dir_from_checkpoint, make_run_dir  # noqa: E402


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and merge any ``inherits:`` parents (deep merge, child wins)."""
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent_name = cfg.pop("inherits", None)
    if parent_name is None:
        return cfg
    parent_path = (path.parent / parent_name).resolve()
    return _deep_merge(load_config(parent_path), cfg)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def bridge_config(raw: dict, in_chans: int) -> dict:
    """Translate the YAML schema (model.*, decoder.*) into ``build_mae`` kwargs."""
    enc = raw.get("model", {})
    dec = raw.get("decoder", {})

    encoder_cfg: dict[str, Any] = {
        "in_chans": in_chans,
        "patch_size": enc.get("patch_size", 16),
        "embed_dim": enc.get("embed_dim", 384),
        "depth": enc.get("depth", 12),
        "num_heads": enc.get("num_heads", 6),
        "mlp_ratio": enc.get("mlp_ratio", 4.0),
        "masking_ratio": enc.get("masking_ratio", 0.75),
        "max_patches": enc.get("max_patches", 1024),
    }

    dec_type = dec.get("type", "transformer")
    decoder_cfg: dict[str, Any] = {
        "type": dec_type,
        "decoder_embed_dim": dec.get("decoder_embed_dim", 512),
        "decoder_num_heads": dec.get("decoder_num_heads", 8),
        "max_patches": dec.get("max_patches", encoder_cfg["max_patches"]),
    }
    if dec_type == "transformer":
        decoder_cfg["decoder_depth"] = dec.get("decoder_depth", 2)
        decoder_cfg["mlp_ratio"] = dec.get("mlp_ratio", 4.0)
    elif dec_type == "kan":
        decoder_cfg["kan_hidden_dim"] = dec.get("kan_hidden_dim", 512)
        decoder_cfg["kan_grid_size"] = dec.get("kan_grid_size", 5)
        decoder_cfg["kan_spline_order"] = dec.get("kan_spline_order", 3)
    else:
        raise ValueError(f"Unknown decoder type {dec_type!r}")

    return {
        "encoder": encoder_cfg,
        "decoder": decoder_cfg,
        "norm_pix_loss": raw.get("norm_pix_loss", True),
    }


def build_dataloaders(raw: dict) -> tuple[DataLoader, DataLoader | None, int]:
    data = raw.get("data", {})
    training = raw.get("training", {})
    name = data.get("dataset", "esc50")
    batch_size = int(training.get("batch_size", 128))
    num_workers = int(data.get("num_workers", 4))

    if name == "esc50":
        from data.audio_dataset import get_esc50_loaders

        train_loader, val_loader, _test_loader = get_esc50_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
            root=data.get("data_root", "data/ESC-50"),
            val_fold=data.get("val_fold", 1),
        )
        return train_loader, val_loader, 1

    if name == "ptbxl":
        from data.ecg_dataset import get_ptbxl_loaders

        train_loader, val_loader, _test_loader = get_ptbxl_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
            root=data.get("data_root", "data/ptbxl"),
            download=bool(data.get("download", False)),
        )
        return train_loader, val_loader, 12

    raise ValueError(f"Unknown dataset {name!r} (expected 'esc50' or 'ptbxl')")


def make_lr_lambda(warmup_steps: int, total_steps: int, min_ratio: float = 0.0):
    """Linear warmup to 1.0, then cosine decay to ``min_ratio``."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_csv_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "step", "train_loss", "lr",
                "band_low", "band_mid", "band_high",
                "epoch_seconds",
            ])


def append_csv_row(path: Path, row: list) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


@torch.no_grad()
def evaluate_band_loss(
    model: MaskedAutoencoder,
    batch: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    """Reconstruct a fixed batch and report low/mid/high-band MSE."""
    was_training = model.training
    model.eval()
    try:
        x = batch.to(device)
        recon, _mask, _grid = model.reconstruct(x)
        bands = MaskedAutoencoder.compute_frequency_band_loss(x, recon)
    finally:
        if was_training:
            model.train()
    return {k: float(v.item()) for k, v in bands.items()}


def fetch_band_eval_batch(loader: DataLoader | None) -> torch.Tensor | None:
    """Pull one batch from the loader and detach it so workers can shut down."""
    if loader is None:
        return None
    try:
        batch = next(iter(loader))
    except StopIteration:
        return None
    x = batch[0] if isinstance(batch, (list, tuple)) else batch
    return x.detach().clone()


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    step: int,
    config: dict,
    loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": config,
            "loss": loss,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, int]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", 0)), int(ckpt.get("step", 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAE pretraining")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size")
    parser.add_argument("--save-every", type=int, default=None, help="Override logging.save_every")
    parser.add_argument("--device", type=str, default=None, help="Override training.device")
    parser.add_argument("--num-workers", type=int, default=None, help="Override data.num_workers")
    parser.add_argument("--seed", type=int, default=None, help="Override training.seed")
    parser.add_argument(
        "--decoder-type",
        type=str,
        default=None,
        choices=["transformer", "kan"],
        help="Override decoder.type",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Explicit run directory. Fresh runs otherwise create a new folder under logging.output_dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw = load_config(args.config)
    if args.epochs is not None:
        raw.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        raw.setdefault("training", {})["batch_size"] = args.batch_size
    if args.device is not None:
        raw.setdefault("training", {})["device"] = args.device
    if args.save_every is not None:
        raw.setdefault("logging", {})["save_every"] = args.save_every
    if args.num_workers is not None:
        raw.setdefault("data", {})["num_workers"] = args.num_workers
    if args.seed is not None:
        raw.setdefault("training", {})["seed"] = args.seed
    if args.decoder_type is not None:
        raw.setdefault("decoder", {})["type"] = args.decoder_type

    seed = int(raw.get("training", {}).get("seed", 42))
    seed_everything(seed)

    train_loader, val_loader, in_chans = build_dataloaders(raw)
    bridged = bridge_config(raw, in_chans=in_chans)
    model = build_mae(bridged)

    device_str = raw.get("training", {}).get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available; falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    model = model.to(device)

    training = raw.get("training", {})
    epochs = int(training.get("epochs", 200))
    lr = float(training.get("lr", 1.5e-4))
    weight_decay = float(training.get("weight_decay", 0.05))
    warmup_epochs = int(training.get("warmup_epochs", 40))
    use_amp = bool(training.get("mixed_precision", True)) and device.type == "cuda"

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * min(warmup_epochs, epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=make_lr_lambda(warmup_steps, total_steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    logging_cfg = raw.get("logging", {})
    base_output_dir = Path(logging_cfg.get("output_dir", "results/"))
    if args.run_dir is not None:
        output_dir = Path(args.run_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    elif args.resume and Path(args.resume).exists():
        output_dir = infer_run_dir_from_checkpoint(args.resume)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = make_run_dir(
            base_output_dir,
            dataset=str(raw.get("data", {}).get("dataset", "dataset")),
            decoder=str(bridged["decoder"]["type"]),
        )
    raw.setdefault("logging", {})["run_dir"] = str(output_dir)

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "training_log.csv"
    init_csv_log(csv_path)
    config_snapshot = output_dir / "config.yaml"
    with config_snapshot.open("w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, sort_keys=False)

    save_every = int(logging_cfg.get("save_every", 20))
    use_wandb = bool(logging_cfg.get("use_wandb", False))

    wandb_run = None
    if use_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=logging_cfg.get("wandb_project", "spectral-kan"),
                config=raw,
                resume="allow",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] wandb init failed: {exc}; continuing without wandb.")
            wandb_run = None

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            start_epoch, global_step = load_checkpoint(
                resume_path, model, optimizer, scheduler, scaler, device
            )
            print(f"resumed from {resume_path} at epoch {start_epoch} (step {global_step})")
        else:
            print(f"[warn] resume path {resume_path} not found; starting fresh.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"device={device} | dataset={raw.get('data', {}).get('dataset')} | "
        f"decoder={bridged['decoder']['type']} | params={n_params:,} | "
        f"epochs={epochs} | batch_size={training.get('batch_size')} | "
        f"steps/epoch={steps_per_epoch} | AMP={'on' if use_amp else 'off'} | "
        f"run_dir={output_dir}"
    )

    band_eval_batch = fetch_band_eval_batch(val_loader or train_loader)

    final_result: dict[str, Any] = {}
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_start = time.time()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", dynamic_ncols=True)
        for batch in pbar:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(x)
                loss = out["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            n_batches += 1
            loss_val = float(loss.detach().item())
            epoch_loss += loss_val
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{current_lr:.2e}")

            if wandb_run is not None:
                wandb_run.log(
                    {"train/loss_step": loss_val, "lr": current_lr, "step": global_step}
                )

        avg_loss = epoch_loss / max(1, n_batches)
        if band_eval_batch is not None:
            bands = evaluate_band_loss(model, band_eval_batch, device)
        else:
            bands = {"low": float("nan"), "mid": float("nan"), "high": float("nan")}
        epoch_seconds = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        final_result = {
            "epoch": epoch + 1,
            "step": global_step,
            "train_loss": avg_loss,
            "lr": current_lr,
            "band_low": bands["low"],
            "band_mid": bands["mid"],
            "band_high": bands["high"],
            "epoch_seconds": round(epoch_seconds, 2),
        }

        append_csv_row(
            csv_path,
            [
                epoch + 1, global_step, avg_loss, current_lr,
                bands["low"], bands["mid"], bands["high"],
                round(epoch_seconds, 2),
            ],
        )

        if wandb_run is not None:
            wandb_run.log({
                "train/loss_epoch": avg_loss,
                "band/low": bands["low"],
                "band/mid": bands["mid"],
                "band/high": bands["high"],
                "epoch": epoch + 1,
            })

        print(
            f"epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f} | "
            f"bands low={bands['low']:.4f} mid={bands['mid']:.4f} high={bands['high']:.4f} | "
            f"{epoch_seconds:.1f}s"
        )

        save_checkpoint(
            ckpt_dir / "last.pt", model, optimizer, scheduler, scaler,
            epoch + 1, global_step, raw, avg_loss,
        )
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt_path = ckpt_dir / "best.pt"
            save_checkpoint(
                best_ckpt_path, model, optimizer, scheduler, scaler,
                epoch + 1, global_step, raw, avg_loss,
            )
            print(f"saved best checkpoint: {best_ckpt_path}")

        is_last_epoch = (epoch + 1) == epochs
        if save_every > 0 and ((epoch + 1) % save_every == 0 or is_last_epoch):
            ckpt_path = ckpt_dir / f"epoch_{epoch + 1:04d}.pt"
            save_checkpoint(
                ckpt_path, model, optimizer, scheduler, scaler,
                epoch + 1, global_step, raw, avg_loss,
            )
            print(f"saved checkpoint: {ckpt_path}")

    if final_result:
        result_path = output_dir / "training_result.yaml"
        with result_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(final_result, f, sort_keys=False)
        print(f"training result written to {result_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
