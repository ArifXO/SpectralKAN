"""train_one_epoch / validate_one_epoch and band-loss helpers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.mae import MaskedAutoencoder  # noqa: E402

from utils.logging_utils import log_wandb_step  # noqa: E402
from utils.metrics import compute_grad_norms  # noqa: E402


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
        out = model(x, return_reconstruction=True)
        bands = MaskedAutoencoder.compute_frequency_band_loss(x, out["recon"])
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


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    epochs: int,
    global_step: int,
    wandb_run: Any = None,
) -> tuple[float, int, dict[str, float]]:
    """Run one training epoch. Returns ``(avg_loss, new_global_step, grad_norms)``.

    ``grad_norms`` are computed from the FINAL batch's grads (still alive
    because ``zero_grad`` only fires at the start of the next batch).
    """
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", dynamic_ncols=True)
    for batch in pbar:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=use_amp):
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

        log_wandb_step(wandb_run, loss_val, current_lr, global_step)

    grad_norms = compute_grad_norms(model)
    avg_loss = epoch_loss / max(1, n_batches)
    return avg_loss, global_step, grad_norms


@torch.no_grad()
def validate_one_epoch(
    model: MaskedAutoencoder,
    val_loader: DataLoader | None,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float] | None:
    """Full val sweep: sample-weighted mean loss + low/mid/high band MSE."""
    if val_loader is None:
        return None

    was_training = model.training
    model.eval()
    total_loss, n_samples = 0.0, 0
    band_acc = {"low": 0.0, "mid": 0.0, "high": 0.0}
    try:
        for batch in val_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device, non_blocking=True)
            bs = x.size(0)
            with torch.amp.autocast(device.type, enabled=use_amp):
                out = model(x, return_reconstruction=True)
            total_loss += float(out["loss"].item()) * bs

            bands = MaskedAutoencoder.compute_frequency_band_loss(x, out["recon"])
            for k, v in bands.items():
                band_acc[k] += float(v.item()) * bs

            n_samples += bs
    finally:
        if was_training:
            model.train()

    if n_samples == 0:
        return None
    return {
        "val_loss": total_loss / n_samples,
        "val_band_low": band_acc["low"] / n_samples,
        "val_band_mid": band_acc["mid"] / n_samples,
        "val_band_high": band_acc["high"] / n_samples,
    }
