"""Periodic feature-tracking + KAN edge-tracking helpers."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.mae import MaskedAutoencoder

from utils.edge_tracker import extract_edge_stats, save_edge_snapshot
from utils.logging_utils import (
    EDGE_LOG_HEADER,
    FEATURE_LOG_HEADER,
    append_csv_row,
    init_csv,
)
from utils.metrics import compute_effective_rank, compute_knn_classification


@torch.no_grad()
def extract_cls_features(
    model: MaskedAutoencoder,
    loader: DataLoader | None,
    device: torch.device,
    max_samples: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run encoder with masking off and collect (CLS feature, label) per sample.

    Multiclass labels are returned as ``(N,)`` and multilabel targets as
    ``(N, C)``. Returns ``None`` if the loader yields no batches.
    """
    if loader is None:
        return None

    was_training = model.training
    model.eval()
    encoder = model.encoder
    feats: list[torch.Tensor] = []
    labs: list[torch.Tensor] = []
    n = 0
    try:
        with encoder.masking_disabled():
            for batch in loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                y = batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else None
                x = x.to(device, non_blocking=True)
                latent, _mask, _ids, _grid = encoder(x)
                feats.append(latent[:, 0].detach().cpu())
                if y is not None:
                    if torch.is_tensor(y):
                        y_cpu = y.detach().cpu()
                    else:
                        y_cpu = torch.as_tensor(y)
                    if y_cpu.dim() > 1:
                        labs.append(y_cpu.float())
                    else:
                        labs.append(y_cpu.long().view(-1))
                n += x.size(0)
                if max_samples is not None and n >= max_samples:
                    break
    finally:
        if was_training:
            model.train()

    if not feats:
        return None
    features = torch.cat(feats, dim=0)
    labels = torch.cat(labs, dim=0) if labs else torch.zeros(features.size(0), dtype=torch.long)
    return features, labels


def run_feature_tracking(
    model: MaskedAutoencoder,
    train_loader: DataLoader,
    test_loader: DataLoader | None,
    device: torch.device,
    epoch: int,
    csv_path: Path,
    feature_max_samples: int | None,
) -> dict[str, float] | None:
    """Compute kNN accuracy + effective rank and append a row to ``csv_path``."""
    train_pack = extract_cls_features(model, train_loader, device, feature_max_samples)
    test_pack = extract_cls_features(model, test_loader, device, feature_max_samples)
    if train_pack is None or test_pack is None:
        return None

    feats_train, labels_train = train_pack
    feats_test, labels_test = test_pack
    is_multilabel = labels_train.dim() > 1 or labels_test.dim() > 1
    task = "multilabel" if is_multilabel else "multiclass"
    metric_name = "macro_auroc" if is_multilabel else "accuracy"
    num_classes = int(labels_train.size(1)) if is_multilabel else None
    knn = compute_knn_classification(
        feats_train,
        labels_train,
        feats_test,
        labels_test,
        k_values=(1, 5),
        task=task,
        num_classes=num_classes,
    )
    eff_train = compute_effective_rank(feats_train)
    eff_test = compute_effective_rank(feats_test)

    row = [
        epoch,
        metric_name,
        round(float(knn[1][metric_name]), 6),
        round(float(knn[5][metric_name]), 6),
        round(float(eff_train), 6),
        round(float(eff_test), 6),
        int(feats_train.size(0)),
        int(feats_test.size(0)),
    ]
    init_csv(csv_path, FEATURE_LOG_HEADER)
    append_csv_row(csv_path, row)
    return {
        "knn_metric": metric_name,
        "knn_k1": knn[1][metric_name],
        "knn_k5": knn[5][metric_name],
        "eff_rank_train": eff_train,
        "eff_rank_test": eff_test,
    }


def run_edge_tracking(
    decoder: nn.Module,
    epoch: int,
    csv_path: Path,
    snapshot_path: Path | None,
) -> dict[str, float] | None:
    """Append an edge-stats row to ``csv_path`` and (optionally) snapshot splines."""
    stats = extract_edge_stats(decoder)
    if not stats:
        return None
    init_csv(csv_path, EDGE_LOG_HEADER)
    append_csv_row(
        csv_path,
        [
            epoch,
            round(float(stats["edge_mean_tv"]), 8),
            round(float(stats["edge_max_tv"]), 8),
            int(stats["n_edges"]),
            int(stats["n_layers"]),
        ],
    )
    if snapshot_path is not None:
        save_edge_snapshot(decoder, snapshot_path)
    return stats
