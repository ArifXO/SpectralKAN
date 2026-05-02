"""Setup helpers: config loading, dataloaders, optimizer/scheduler, seeds, run dirs."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


from utils.output_paths import infer_run_dir_from_checkpoint, make_run_dir


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


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


def bridge_config(raw: dict, in_chans: int) -> dict:
    """Translate the YAML schema (model.*, decoder.*) into ``build_mae`` kwargs."""
    enc = raw.get("model", {})
    dec = raw.get("decoder", {})

    encoder_type = enc.get("encoder_type", "vit_small")
    if encoder_type != "vit_small":
        raise ValueError(
            f"Unsupported encoder_type={encoder_type!r}. "
            "This project only supports the ViT-Small encoder."
        )

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


def validate_project_config(raw: dict, device: torch.device | None = None) -> None:
    """Enforce project-level constraints from AGENTS.md before expensive work starts."""
    enc = raw.get("model", {})
    encoder_type = enc.get("encoder_type", "vit_small")
    if encoder_type != "vit_small":
        raise ValueError(
            f"Unsupported encoder_type={encoder_type!r}. Do not use ViT-Large in this project."
        )

    masking_ratio = float(enc.get("masking_ratio", 0.75))
    if not math.isclose(masking_ratio, 0.75, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"Unsupported masking_ratio={masking_ratio}. "
            "This project keeps MAE masking fixed at 0.75."
        )

    training = raw.get("training", {})
    batch_size = int(training.get("batch_size", 128))
    max_batch = _batch_limit_for_device(device)
    if batch_size > max_batch:
        device_desc = str(device) if device is not None else "configured device"
        raise ValueError(
            f"batch_size={batch_size} exceeds the project limit of {max_batch} for {device_desc}."
        )


def _batch_limit_for_device(device: torch.device | None) -> int:
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        name = torch.cuda.get_device_name(device).lower()
        if "3060" in name:
            return 128
        if "4070" in name:
            return 256
    return 256


def validate_checkpoint_dataset(
    checkpoint_config: dict,
    requested_dataset: str,
    checkpoint_path: str | Path,
) -> None:
    """Fail fast when a checkpoint belongs to a different dataset than requested."""
    ckpt_dataset = checkpoint_config.get("data", {}).get("dataset")
    if ckpt_dataset is not None and ckpt_dataset != requested_dataset:
        raise ValueError(
            f"Checkpoint {checkpoint_path} was trained for dataset={ckpt_dataset!r}, "
            f"but dataset={requested_dataset!r} was requested."
        )


def validate_resume_compatible(current_raw: dict, checkpoint_raw: dict, in_chans: int) -> None:
    """Ensure the active config builds the same MAE architecture as the checkpoint."""
    if not checkpoint_raw:
        return
    current = bridge_config(current_raw, in_chans=in_chans)
    saved = bridge_config(checkpoint_raw, in_chans=in_chans)
    if current != saved:
        raise ValueError(
            "Resume checkpoint architecture does not match the active config. "
            "Use the original config or matching decoder/model overrides."
        )


def build_dataloaders(
    raw: dict,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None, int]:
    """Build train/val/test loaders and return the input-channel count."""
    data = raw.get("data", {})
    training = raw.get("training", {})
    name = data.get("dataset", "esc50")
    batch_size = int(training.get("batch_size", 128))
    num_workers = int(data.get("num_workers", 4))

    if name == "esc50":
        from data.audio_dataset import get_esc50_loaders

        train_loader, val_loader, test_loader = get_esc50_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
            root=data.get("data_root", "data/ESC-50"),
            val_fold=data.get("val_fold", 4),
        )
        return train_loader, val_loader, test_loader, 1

    if name == "ptbxl":
        from data.ecg_dataset import get_ptbxl_loaders

        train_loader, val_loader, test_loader = get_ptbxl_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
            root=data.get("data_root", "data/ptbxl"),
            download=bool(data.get("download", False)),
        )
        return train_loader, val_loader, test_loader, 12

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


def build_optimizer(model: torch.nn.Module, training_cfg: dict) -> torch.optim.Optimizer:
    """AdamW with the He et al. 2022 MAE betas."""
    lr = float(training_cfg.get("lr", 1.5e-4))
    weight_decay = float(training_cfg.get("weight_decay", 0.05))
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_cfg: dict,
    steps_per_epoch: int,
    epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup + cosine decay LambdaLR."""
    warmup_epochs = int(training_cfg.get("warmup_epochs", 40))
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * min(warmup_epochs, epochs)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=make_lr_lambda(warmup_steps, total_steps)
    )


def get_run_dir_name(
    base_output_dir: Path,
    run_dir_arg: str | None,
    resume_arg: str | None,
    dataset: str,
    decoder: str,
) -> Path:
    """Resolve the run directory: explicit override, resume, or fresh timestamped folder."""
    if run_dir_arg is not None:
        output_dir = Path(run_dir_arg)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    if resume_arg and Path(resume_arg).exists():
        output_dir = infer_run_dir_from_checkpoint(resume_arg)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    return make_run_dir(base_output_dir, dataset=dataset, decoder=decoder)
