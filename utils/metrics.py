"""Pure metric helpers shared between training, evaluation, and analysis.

These functions never touch the filesystem and never mutate model parameters,
so any caller is free to decide where the numbers go (CSV, wandb, stdout).
The one exception is :func:`get_gpu_memory_mb`, which by contract resets the
peak-memory counter after reading.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_band_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_bands: int = 3,
    freq_dim: int = -2,
) -> dict[str, float]:
    """Per-band MSE along the spectrogram frequency axis.

    Spectrograms in this project are ``(B, C, H, W)`` with frequency on H, so
    the default ``freq_dim=-2`` matches them. For ``n_bands == 3`` the keys
    are ``band_low / band_mid / band_high``; otherwise they are
    ``band_0 .. band_{n_bands-1}``.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    if n_bands == 3:
        names = ["band_low", "band_mid", "band_high"]
    else:
        names = [f"band_{i}" for i in range(n_bands)]
    return {
        key: float(value.item())
        for key, value in compute_band_mse_tensors(
            pred, target, n_bands=n_bands, freq_dim=freq_dim, names=names
        ).items()
    }


def compute_band_mse_tensors(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_bands: int = 3,
    freq_dim: int = -2,
    names: Sequence[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Tensor-returning variant used by training code that logs torch metrics."""
    if pred.shape != target.shape:
        raise ValueError(
            f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    if n_bands < 1:
        raise ValueError(f"n_bands must be >= 1, got {n_bands}")
    f_size = pred.size(freq_dim)
    if f_size < n_bands:
        raise ValueError(
            f"need at least {n_bands} bins on freq_dim, got {f_size}"
        )

    if names is None:
        names = [f"band_{i}" for i in range(n_bands)]
    if len(names) != n_bands:
        raise ValueError(f"names length must match n_bands ({n_bands})")

    edges = [(i * f_size) // n_bands for i in range(n_bands + 1)]
    out: dict[str, torch.Tensor] = {}
    for i, name in enumerate(names):
        sl: list[slice] = [slice(None)] * pred.dim()
        sl[freq_dim] = slice(edges[i], edges[i + 1])
        sl_t = tuple(sl)
        out[name] = F.mse_loss(pred[sl_t], target[sl_t])
    return out


def compute_grad_norms(
    model: nn.Module,
    prefix_groups: Mapping[str, str] | None = None,
) -> dict[str, float]:
    """Total L2 gradient norm per named parameter group.

    Each entry in ``prefix_groups`` maps a group label to a parameter-name
    prefix; the result keys are ``grad_norm_{label}``. Parameters whose
    ``.grad`` is ``None`` are skipped, and a group with no matching grads
    reports ``0.0``.
    """
    if prefix_groups is None:
        prefix_groups = {"encoder": "encoder", "decoder": "decoder"}

    out: dict[str, float] = {}
    for group_name, prefix in prefix_groups.items():
        sq_sum = 0.0
        for name, param in model.named_parameters():
            if param.grad is None or not name.startswith(prefix):
                continue
            sq_sum += float(param.grad.detach().pow(2).sum().item())
        out[f"grad_norm_{group_name}"] = math.sqrt(sq_sum)
    return out


def compute_knn_classification(
    features_train: torch.Tensor,
    labels_train: torch.Tensor,
    features_test: torch.Tensor,
    labels_test: torch.Tensor,
    k_values: Sequence[int] = (1, 5),
    task: str = "multiclass",
    num_classes: int | None = None,
    device: torch.device | str | None = None,
    chunk_size: int = 1024,
) -> dict[int, dict[str, float]]:
    """Cosine-similarity kNN metrics for multiclass or multilabel labels."""
    if features_train.dim() != 2 or features_test.dim() != 2:
        raise ValueError("features must be 2D (N, D)")
    if features_train.size(1) != features_test.size(1):
        raise ValueError("train/test feature dims differ")
    if features_train.size(0) != labels_train.size(0):
        raise ValueError("train features/labels length mismatch")
    if features_test.size(0) != labels_test.size(0):
        raise ValueError("test features/labels length mismatch")

    max_k = max(k_values)
    if max_k > features_train.size(0):
        raise ValueError(f"k={max_k} > number of train samples")
    if task not in {"multiclass", "multilabel"}:
        raise ValueError(f"task must be 'multiclass' or 'multilabel', got {task!r}")

    top_idx = _cosine_topk_indices(
        features_train,
        features_test,
        max_k=max_k,
        device=device,
        chunk_size=chunk_size,
    )

    out: dict[int, dict[str, float]] = {}
    if task == "multiclass":
        labels_train_l = labels_train.detach().cpu().long().view(-1)
        labels_test_l = labels_test.detach().cpu().long().view(-1)
        if num_classes is None:
            num_classes = int(
                max(labels_train_l.max().item(), labels_test_l.max().item())
            ) + 1

        top_labels = labels_train_l[top_idx]
        for k in k_values:
            k_labels = top_labels[:, :k]
            votes = F.one_hot(k_labels, num_classes=num_classes).sum(dim=1)
            preds = votes.argmax(dim=1)
            acc = (preds == labels_test_l.to(preds.device)).float().mean().item()
            out[int(k)] = {"accuracy": float(acc)}
        return out

    neighbour_labels = labels_train.detach().cpu().float()[top_idx]
    targets = labels_test.detach().cpu().float().numpy()
    try:
        from sklearn.metrics import roc_auc_score
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("scikit-learn is required for multilabel kNN AUROC") from exc

    for k in k_values:
        scores = neighbour_labels[:, :k].mean(dim=1).numpy()
        try:
            auroc = float(roc_auc_score(targets, scores, average="macro"))
        except ValueError:
            auroc = float("nan")
        out[int(k)] = {"macro_auroc": auroc}
    return out


def _cosine_topk_indices(
    features_train: torch.Tensor,
    features_test: torch.Tensor,
    max_k: int,
    device: torch.device | str | None = None,
    chunk_size: int = 1024,
) -> torch.Tensor:
    target_device = torch.device(device) if device is not None else features_train.device
    train_norm = F.normalize(features_train.float(), dim=1).to(target_device)
    test_norm = F.normalize(features_test.float(), dim=1).to(target_device)

    chunks: list[torch.Tensor] = []
    for i in range(0, test_norm.size(0), chunk_size):
        sims = test_norm[i : i + chunk_size] @ train_norm.t()
        chunks.append(sims.topk(max_k, dim=1).indices.cpu())
    return torch.cat(chunks, dim=0)


def compute_effective_rank(feature_matrix: torch.Tensor) -> float:
    """Effective rank = ``exp(H(p))`` where ``p`` is the normalised SVD spectrum.

    A rank-1 matrix yields 1.0; a perfectly isotropic ``N x D`` matrix yields
    ``min(N, D)``. Useful for tracking representation collapse.
    """
    if feature_matrix.dim() != 2:
        raise ValueError("feature_matrix must be 2D")
    s = torch.linalg.svdvals(feature_matrix.float())
    s = s[s > 1e-12]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    entropy = -(p * p.log()).sum().item()
    return math.exp(entropy)


def count_parameters(model: nn.Module, prefix: str = "") -> int:
    """Number of trainable parameters whose ``named_parameters`` name starts with ``prefix``."""
    total = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if prefix and not name.startswith(prefix):
            continue
        total += param.numel()
    return total


def estimate_flops(
    model: nn.Module,
    dummy_input: torch.Tensor | tuple,
) -> int | None:
    """Forward-pass FLOPs estimate via ``torch.utils.flop_counter`` or ``thop``.

    Returns ``None`` if neither backend can run on this model. Never raises.
    The model is briefly switched to ``eval`` mode and restored.
    """
    args = dummy_input if isinstance(dummy_input, tuple) else (dummy_input,)

    try:
        from torch.utils.flop_counter import FlopCounterMode  # type: ignore

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                with FlopCounterMode(display=False) as counter:
                    model(*args)
            return int(counter.get_total_flops())
        finally:
            if was_training:
                model.train()
    except Exception:
        pass

    try:
        from thop import profile  # type: ignore

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                flops, _params = profile(model, inputs=args, verbose=False)
            return int(flops)
        finally:
            if was_training:
                model.train()
    except Exception:
        pass

    return None


def get_gpu_memory_mb() -> float:
    """Peak CUDA memory in MB since the last reset; resets the counter on read.

    Returns ``0.0`` if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return 0.0
    peak_bytes = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    return peak_bytes / (1024 ** 2)
