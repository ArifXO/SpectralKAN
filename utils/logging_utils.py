"""CSV + wandb logging helpers for training runs."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


TRAINING_LOG_HEADER = [
    "epoch", "train_loss", "val_loss",
    "band_low", "band_mid", "band_high",
    "val_band_low", "val_band_mid", "val_band_high",
    "lr", "epoch_seconds",
    "grad_norm_encoder", "grad_norm_decoder",
    "peak_gpu_mb",
]

FEATURE_LOG_HEADER = [
    "epoch", "knn_metric", "knn_k1", "knn_k5",
    "eff_rank_train", "eff_rank_test",
    "n_train", "n_test",
]

EDGE_LOG_HEADER = ["epoch", "edge_mean_tv", "edge_max_tv", "n_edges", "n_layers"]


def init_csv(path: Path, header: list[str]) -> None:
    """Create ``path`` (and its parent dirs) with a header row if it doesn't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def append_csv_row(path: Path, row: list) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def log_epoch_row(
    csv_path: Path,
    epoch: int,
    avg_loss: float,
    val_loss: float,
    bands: dict[str, float],
    val_band_low: float,
    val_band_mid: float,
    val_band_high: float,
    current_lr: float,
    epoch_seconds: float,
    grad_norms: dict[str, float],
    peak_gpu: float,
) -> None:
    """Append one training-log row to ``csv_path`` matching ``TRAINING_LOG_HEADER``."""
    append_csv_row(
        csv_path,
        [
            epoch,
            round(avg_loss, 6),
            round(val_loss, 6) if not math.isnan(val_loss) else "",
            round(bands["low"], 6), round(bands["mid"], 6), round(bands["high"], 6),
            round(val_band_low, 6) if not math.isnan(val_band_low) else "",
            round(val_band_mid, 6) if not math.isnan(val_band_mid) else "",
            round(val_band_high, 6) if not math.isnan(val_band_high) else "",
            current_lr,
            round(epoch_seconds, 2),
            round(grad_norms.get("grad_norm_encoder", 0.0), 6),
            round(grad_norms.get("grad_norm_decoder", 0.0), 6),
            round(peak_gpu, 2),
        ],
    )


def init_wandb(logging_cfg: dict, raw_snapshot: dict) -> Any:
    """Initialise a wandb run if ``logging.use_wandb`` is True; ``None`` on failure."""
    if not bool(logging_cfg.get("use_wandb", False)):
        return None
    try:
        import wandb

        return wandb.init(
            project=logging_cfg.get("wandb_project", "spectral-kan"),
            config=raw_snapshot,
            resume="allow",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] wandb init failed: {exc}; continuing without wandb.")
        return None


def log_wandb_epoch(
    wandb_run: Any,
    avg_loss: float,
    val_loss: float,
    bands: dict[str, float],
    val_band_low: float,
    val_band_mid: float,
    val_band_high: float,
    grad_norms: dict[str, float],
    peak_gpu: float,
    epoch: int,
) -> None:
    if wandb_run is None:
        return
    wandb_run.log({
        "train/loss_epoch": avg_loss,
        "val/loss_epoch": val_loss,
        "band/low": bands["low"], "band/mid": bands["mid"], "band/high": bands["high"],
        "val_band/low": val_band_low, "val_band/mid": val_band_mid, "val_band/high": val_band_high,
        "grad_norm/encoder": grad_norms.get("grad_norm_encoder", 0.0),
        "grad_norm/decoder": grad_norms.get("grad_norm_decoder", 0.0),
        "peak_gpu_mb": peak_gpu,
        "epoch": epoch,
    })


def log_wandb_step(wandb_run: Any, loss_val: float, current_lr: float, global_step: int) -> None:
    if wandb_run is None:
        return
    wandb_run.log({"train/loss_step": loss_val, "lr": current_lr, "step": global_step})
