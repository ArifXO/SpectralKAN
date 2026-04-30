"""Linear probe + k-NN evaluation for MAE-pretrained encoders.

Usage:
    python scripts/evaluate.py --checkpoint results/<run>/checkpoints/last.pt --dataset esc50
    python scripts/evaluate.py --checkpoint results/<run>/checkpoints/last.pt --dataset ptbxl

Loads an MAE checkpoint, freezes the encoder, extracts cls-token features once
for the train and test splits, then evaluates two probes on top:

* Linear probe: a single ``nn.Linear`` head trained for ``--linear-epochs`` (default 100)
  with SGD(lr=0.1, momentum=0.9) + cosine decay. CrossEntropy for ESC-50
  (single-label, 50 classes) → top-1 accuracy. BCEWithLogits for PTB-XL
  (multi-label, 5 superclasses) → macro-AUROC.
* k-NN: cosine-similarity nearest neighbours in feature space, no training.
  Defaults to k=1 and k=5. Same metrics as the linear probe (ESC-50: majority
  vote → accuracy; PTB-XL: averaged neighbour labels → macro-AUROC).

Results are appended under the checkpoint's run folder by default:
``results/<run>/evaluation/eval_results.csv``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.mae import build_mae  # noqa: E402
from output_paths import infer_run_dir_from_checkpoint  # noqa: E402
from train import bridge_config  # noqa: E402  -- shared YAML→build_mae translator


def load_eval_loaders(
    dataset: str, batch_size: int, num_workers: int, data_root: str
) -> tuple[DataLoader, DataLoader, int, int, str]:
    """Return ``(train_loader, test_loader, in_chans, num_classes, task)``.

    Linear probes evaluate downstream classification, so we use the standard
    train/test folds for each dataset and ignore the val split.
    """
    if dataset == "esc50":
        from data.audio_dataset import get_esc50_loaders

        train_loader, _val, test_loader = get_esc50_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
            root=data_root,
            val_fold=None,
        )
        return train_loader, test_loader, 1, 50, "multiclass"

    if dataset == "ptbxl":
        from data.ecg_dataset import get_ptbxl_loaders

        train_loader, _val, test_loader = get_ptbxl_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
            root=data_root,
            download=False,
        )
        return train_loader, test_loader, 12, 5, "multilabel"

    raise ValueError(f"Unknown dataset {dataset!r} (expected 'esc50' or 'ptbxl')")


@torch.no_grad()
def extract_features(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the encoder with masking disabled and stack cls-token features."""
    encoder.eval()
    feats: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    with encoder.masking_disabled():
        for batch in tqdm(loader, desc="features", dynamic_ncols=True):
            x, y = batch
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                latent, _mask, _ids, _grid = encoder(x)
            feats.append(latent[:, 0].float().cpu())
            labels.append(y if isinstance(y, torch.Tensor) else torch.as_tensor(y))

    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def linear_probe(
    f_tr: torch.Tensor,
    y_tr: torch.Tensor,
    f_te: torch.Tensor,
    y_te: torch.Tensor,
    num_classes: int,
    task: str,
    lr: float,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    head = nn.Linear(f_tr.size(1), num_classes).to(device)
    optimizer = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss() if task == "multiclass" else nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(f_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    head.train()
    pbar = tqdm(range(epochs), desc="linear probe", dynamic_ncols=True)
    for _epoch in pbar:
        running = 0.0
        for fb, yb in train_loader:
            fb = fb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            yb = yb.long() if task == "multiclass" else yb.float()
            optimizer.zero_grad()
            loss = criterion(head(fb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * fb.size(0)
        scheduler.step()
        pbar.set_postfix(
            loss=f"{running / len(train_ds):.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    head.eval()
    with torch.no_grad():
        logits = head(f_te.to(device)).cpu()

    if task == "multiclass":
        preds = logits.argmax(dim=1)
        return {"accuracy": float((preds == y_te.long()).float().mean().item())}

    scores = torch.sigmoid(logits).numpy()
    targets = y_te.numpy()
    try:
        auroc = float(roc_auc_score(targets, scores, average="macro"))
    except ValueError:
        auroc = float("nan")
    return {"macro_auroc": auroc}


def knn_eval(
    f_tr: torch.Tensor,
    y_tr: torch.Tensor,
    f_te: torch.Tensor,
    y_te: torch.Tensor,
    k: int,
    num_classes: int,
    task: str,
    device: torch.device,
) -> dict[str, float]:
    """Cosine-similarity k-NN. Multiclass: majority vote. Multilabel: averaged labels."""
    f_tr_n = F.normalize(f_tr, dim=1).to(device)
    f_te_n = F.normalize(f_te, dim=1).to(device)

    chunk = 1024
    chunks: list[torch.Tensor] = []
    for i in range(0, f_te_n.size(0), chunk):
        sims = f_te_n[i : i + chunk] @ f_tr_n.T
        chunks.append(sims.topk(k, dim=1).indices.cpu())
    top_idx = torch.cat(chunks, dim=0)

    if task == "multiclass":
        neighbour_labels = y_tr.long()[top_idx]  # (N_test, k)
        preds = torch.empty(neighbour_labels.size(0), dtype=torch.long)
        for i in range(neighbour_labels.size(0)):
            preds[i] = torch.bincount(neighbour_labels[i], minlength=num_classes).argmax()
        return {"accuracy": float((preds == y_te.long()).float().mean().item())}

    neighbour_labels = y_tr.float()[top_idx]  # (N_test, k, C)
    scores = neighbour_labels.mean(dim=1).numpy()
    targets = y_te.numpy()
    try:
        auroc = float(roc_auc_score(targets, scores, average="macro"))
    except ValueError:
        auroc = float("nan")
    return {"macro_auroc": auroc}


def append_results(path: Path, rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(
                ["timestamp", "checkpoint", "dataset", "method", "metric", "value"]
            )
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAE encoder linear probe + k-NN")
    parser.add_argument("--checkpoint", required=True, help="Path to MAE checkpoint .pt")
    parser.add_argument("--dataset", required=True, choices=["esc50", "ptbxl"])
    parser.add_argument("--data-root", default=None, help="Override data root from config")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--linear-epochs", type=int, default=100)
    parser.add_argument("--linear-lr", type=float, default=0.1)
    parser.add_argument("--linear-batch-size", type=int, default=256)
    parser.add_argument("--knn-k", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        default=None,
        help="CSV output path. Defaults to <checkpoint-run>/evaluation/eval_results.csv",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from train import seed_everything

    seed_everything(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    checkpoint_path = Path(args.checkpoint)
    output_path = (
        Path(args.output)
        if args.output is not None
        else infer_run_dir_from_checkpoint(checkpoint_path) / "evaluation" / "eval_results.csv"
    )

    print(f"loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    raw_cfg = ckpt.get("config", {})
    if not raw_cfg:
        print("[warn] checkpoint has no 'config'; assuming default model dims.")

    in_chans = 1 if args.dataset == "esc50" else 12
    bridged = bridge_config(raw_cfg, in_chans=in_chans)

    model = build_mae(bridged)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(
            f"[warn] state_dict load: missing={len(missing)} unexpected={len(unexpected)}"
        )

    encoder = model.encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad = False

    data_root = args.data_root or (
        "data/ESC-50" if args.dataset == "esc50" else "data/ptbxl"
    )
    train_loader, test_loader, _ic, num_classes, task = load_eval_loaders(
        args.dataset, args.batch_size, args.num_workers, data_root
    )

    use_amp = (device.type == "cuda") and not args.no_amp
    print(
        f"device={device} | dataset={args.dataset} | task={task} | "
        f"num_classes={num_classes} | AMP={'on' if use_amp else 'off'}"
    )

    print("extracting train features...")
    f_tr, y_tr = extract_features(encoder, train_loader, device, use_amp)
    print("extracting test features...")
    f_te, y_te = extract_features(encoder, test_loader, device, use_amp)
    print(f"train features: {tuple(f_tr.shape)}  labels: {tuple(y_tr.shape)}")
    print(f"test  features: {tuple(f_te.shape)}  labels: {tuple(y_te.shape)}")

    rows: list[list] = []
    timestamp = datetime.now().isoformat(timespec="seconds")

    print("running linear probe...")
    lp = linear_probe(
        f_tr, y_tr, f_te, y_te, num_classes, task,
        lr=args.linear_lr, epochs=args.linear_epochs,
        batch_size=args.linear_batch_size, device=device,
    )
    for metric, value in lp.items():
        rows.append([timestamp, str(checkpoint_path), args.dataset, "linear_probe", metric, f"{value:.6f}"])
        print(f"  linear_probe {metric}: {value:.4f}")

    for k in args.knn_k:
        print(f"running {k}-NN...")
        kn = knn_eval(
            f_tr, y_tr, f_te, y_te,
            k=k, num_classes=num_classes, task=task, device=device,
        )
        for metric, value in kn.items():
            rows.append([timestamp, str(checkpoint_path), args.dataset, f"knn{k}", metric, f"{value:.6f}"])
            print(f"  {k}-NN {metric}: {value:.4f}")

    append_results(output_path, rows)
    print(f"results appended to {output_path}")


if __name__ == "__main__":
    main()
