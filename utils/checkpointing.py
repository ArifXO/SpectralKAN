"""Checkpoint save/load and run-summary helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml


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
    best_val_loss: float = float("inf"),
    best_val_epoch: int = 0,
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
            "best_val_loss": best_val_loss,
            "best_val_epoch": best_val_epoch,
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
) -> tuple[int, int, float, int]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return (
        int(ckpt.get("epoch", 0)),
        int(ckpt.get("step", 0)),
        float(ckpt.get("best_val_loss", float("inf"))),
        int(ckpt.get("best_val_epoch", 0)),
    )


def load_checkpoint_config(path: Path, device: torch.device | str = "cpu") -> dict:
    """Read the saved training config from a checkpoint, returning ``{}`` if absent."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = ckpt.get("config", {})
    return cfg if isinstance(cfg, dict) else {}


def write_run_summary(
    output_dir: Path,
    raw: dict,
    decoder_type: str,
    epochs: int,
    last_train_loss: float,
    last_val_loss: float,
    best_val_loss: float,
    best_val_epoch: int,
    encoder_params: int,
    decoder_params: int,
    total_params: int,
    estimated_flops: int | None,
    device: torch.device,
    total_train_seconds: float,
    seed: int,
    config_path: Path,
    global_step: int,
) -> None:
    """Write the human-readable ``run_summary.txt`` and ``training_result.yaml`` files."""
    summary_lines = [
        f"dataset:           {raw.get('data', {}).get('dataset')}",
        f"decoder type:      {decoder_type}",
        f"total epochs:      {epochs}",
        f"final train_loss:  {last_train_loss:.6f}",
        f"final val_loss:    {last_val_loss:.6f}" if not math.isnan(last_val_loss) else "final val_loss:    n/a",
        f"best val_loss:     {best_val_loss:.6f} @ epoch {best_val_epoch}"
        if best_val_loss != float("inf") else "best val_loss:     n/a",
        f"encoder params:    {encoder_params:,}",
        f"decoder params:    {decoder_params:,}",
        f"total params:      {total_params:,}",
        f"estimated FLOPs:   {estimated_flops:,}" if estimated_flops is not None else "estimated FLOPs:   n/a",
        f"device:            {device}",
        f"total train time:  {total_train_seconds / 60:.2f} min ({total_train_seconds:.1f}s)",
        f"seed:              {seed}",
        f"config file:       {Path(config_path).resolve()}",
        f"run dir:           {output_dir}",
    ]
    summary_path = output_dir / "run_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nrun summary written to {summary_path}")

    final_result: dict[str, Any] = {
        "epoch": epochs,
        "step": global_step,
        "train_loss": last_train_loss,
        "val_loss": last_val_loss,
        "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
        "best_val_epoch": best_val_epoch,
        "total_train_seconds": round(total_train_seconds, 2),
    }
    result_path = output_dir / "training_result.yaml"
    with result_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(final_result, f, sort_keys=False)
    print(f"training result written to {result_path}")
