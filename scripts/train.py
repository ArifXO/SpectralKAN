"""MAE pretraining entry point.

Usage:
    python scripts/train.py --config configs/esc50_config.yaml
    python scripts/train.py --config configs/ptbxl_config.yaml --resume results/<run>/checkpoints/last.pt
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.mae import build_mae  # noqa: E402

from utils.checkpointing import (  # noqa: E402
    load_checkpoint,
    load_checkpoint_config,
    save_checkpoint,
    write_run_summary,
)
from utils.logging_utils import (  # noqa: E402
    TRAINING_LOG_HEADER,
    init_csv,
    init_wandb,
    log_epoch_row,
    log_wandb_epoch,
)
from utils.metrics import count_parameters, estimate_flops, get_gpu_memory_mb  # noqa: E402
from utils.periodic_evals import run_edge_tracking, run_feature_tracking  # noqa: E402
from utils.setup import (  # noqa: E402
    bridge_config,
    build_dataloaders,
    build_optimizer,
    build_scheduler,
    get_run_dir_name,
    load_config,
    seed_everything,
    validate_project_config,
    validate_resume_compatible,
)
from utils.training_loop import (  # noqa: E402
    evaluate_band_loss,
    fetch_band_eval_batch,
    train_one_epoch,
    validate_one_epoch,
)


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
        "--decoder-type", type=str, default=None, choices=["transformer", "kan"],
        help="Override decoder.type",
    )
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Explicit run directory. Fresh runs otherwise create a new folder under logging.output_dir.",
    )
    return parser.parse_args()


def apply_cli_overrides(raw: dict, args: argparse.Namespace) -> None:
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


def main() -> None:
    args = parse_args()
    raw = load_config(args.config)
    apply_cli_overrides(raw, args)

    seed = int(raw.get("training", {}).get("seed", 42))
    seed_everything(seed)

    training = raw.get("training", {})
    device_str = training.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available; falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    validate_project_config(raw, device)

    resume_config: dict = {}
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            resume_config = load_checkpoint_config(resume_path, device="cpu")

    train_loader, val_loader, test_loader, in_chans = build_dataloaders(raw)
    if resume_config:
        validate_resume_compatible(raw, resume_config, in_chans=in_chans)
    bridged = bridge_config(raw, in_chans=in_chans)
    model = build_mae(bridged)

    model = model.to(device)

    epochs = int(training.get("epochs", 200))
    use_amp = bool(training.get("mixed_precision", True)) and device.type == "cuda"
    optimizer = build_optimizer(model, training)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = build_scheduler(optimizer, training, steps_per_epoch, epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    logging_cfg = raw.get("logging", {})
    base_output_dir = Path(logging_cfg.get("output_dir", "results/"))
    output_dir = get_run_dir_name(
        base_output_dir, args.run_dir, args.resume,
        dataset=str(raw.get("data", {}).get("dataset", "dataset")),
        decoder=str(bridged["decoder"]["type"]),
    )
    raw.setdefault("logging", {})["run_dir"] = str(output_dir)

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = output_dir / "edge_snapshots"
    csv_path = output_dir / "training_log.csv"
    feature_csv_path = output_dir / "feature_tracking.csv"
    edge_csv_path = output_dir / "edge_evolution.csv"
    init_csv(csv_path, TRAINING_LOG_HEADER)

    save_every = int(logging_cfg.get("save_every", 20))
    feature_every = int(logging_cfg.get("feature_every", 25))
    edge_every = int(logging_cfg.get("edge_every", 25))
    edge_snapshot_every = int(logging_cfg.get("edge_snapshot_every", 50))
    feature_max_samples = logging_cfg.get("feature_max_samples", None)
    if feature_max_samples is not None:
        feature_max_samples = int(feature_max_samples)

    encoder_params = count_parameters(model, prefix="encoder")
    decoder_params = count_parameters(model, prefix="decoder")
    total_params = count_parameters(model)
    try:
        sample_batch = next(iter(train_loader))
        sample_x = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch
        dummy = sample_x[:1].to(device)
        estimated_flops = estimate_flops(model, dummy)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] FLOP estimate skipped: {exc}")
        estimated_flops = None

    raw_snapshot = dict(raw)
    raw_snapshot["computed"] = {
        "encoder_params": int(encoder_params),
        "decoder_params": int(decoder_params),
        "total_params": int(total_params),
        "estimated_flops": int(estimated_flops) if estimated_flops is not None else None,
        "device": str(device),
        "config_file": str(Path(args.config).resolve()),
    }
    config_snapshot = output_dir / "config.yaml"
    with config_snapshot.open("w", encoding="utf-8") as f:
        yaml.safe_dump(raw_snapshot, f, sort_keys=False)

    wandb_run = init_wandb(logging_cfg, raw_snapshot)

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    best_val_epoch = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            start_epoch, global_step, best_val_loss, best_val_epoch = load_checkpoint(
                resume_path, model, optimizer, scheduler, scaler, device
            )
            print(
                f"resumed from {resume_path} at epoch {start_epoch} (step {global_step}); "
                f"best val_loss={best_val_loss:.4f} @ epoch {best_val_epoch}"
            )
        else:
            print(f"[warn] resume path {resume_path} not found; starting fresh.")

    print(
        f"device={device} | dataset={raw.get('data', {}).get('dataset')} | "
        f"decoder={bridged['decoder']['type']} | params={total_params:,} "
        f"(enc={encoder_params:,} dec={decoder_params:,}) | "
        f"epochs={epochs} | batch_size={training.get('batch_size')} | "
        f"steps/epoch={steps_per_epoch} | AMP={'on' if use_amp else 'off'} | "
        f"run_dir={output_dir}"
    )

    decoder_type = bridged["decoder"]["type"]
    band_eval_batch = fetch_band_eval_batch(val_loader or train_loader)
    train_start_wall = time.time()
    last_train_loss = float("nan")
    last_val_loss = float("nan")

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        avg_loss, global_step, grad_norms = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, use_amp, epoch, epochs, global_step, wandb_run,
        )
        last_train_loss = avg_loss

        if band_eval_batch is not None:
            bands = evaluate_band_loss(model, band_eval_batch, device)
        else:
            bands = {"low": float("nan"), "mid": float("nan"), "high": float("nan")}

        val_metrics = validate_one_epoch(model, val_loader, device, use_amp)
        if val_metrics is not None:
            val_loss = val_metrics["val_loss"]
            val_band_low = val_metrics["val_band_low"]
            val_band_mid = val_metrics["val_band_mid"]
            val_band_high = val_metrics["val_band_high"]
            last_val_loss = val_loss
        else:
            val_loss = float("nan")
            val_band_low = val_band_mid = val_band_high = float("nan")

        peak_gpu = get_gpu_memory_mb()
        epoch_seconds = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        log_epoch_row(
            csv_path, epoch + 1, avg_loss, val_loss, bands,
            val_band_low, val_band_mid, val_band_high,
            current_lr, epoch_seconds, grad_norms, peak_gpu,
        )
        log_wandb_epoch(
            wandb_run, avg_loss, val_loss, bands,
            val_band_low, val_band_mid, val_band_high,
            grad_norms, peak_gpu, epoch + 1,
        )

        val_loss_str = f"{val_loss:.4f}" if not math.isnan(val_loss) else "n/a"
        print(
            f"epoch {epoch + 1}/{epochs}: train={avg_loss:.4f} val={val_loss_str} | "
            f"bands low={bands['low']:.4f} mid={bands['mid']:.4f} high={bands['high']:.4f} | "
            f"grad enc={grad_norms.get('grad_norm_encoder', 0.0):.3f} "
            f"dec={grad_norms.get('grad_norm_decoder', 0.0):.3f} | "
            f"gpu={peak_gpu:.0f}MB | {epoch_seconds:.1f}s"
        )

        improved = (not math.isnan(val_loss)) and val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_val_epoch = epoch + 1
            save_checkpoint(
                ckpt_dir / "best_model.pt", model, optimizer, scheduler, scaler,
                epoch + 1, global_step, raw, avg_loss,
                best_val_loss=best_val_loss, best_val_epoch=best_val_epoch,
            )
            print(f"saved best_model.pt (val_loss={best_val_loss:.4f})")

        save_checkpoint(
            ckpt_dir / "last.pt", model, optimizer, scheduler, scaler,
            epoch + 1, global_step, raw, avg_loss,
            best_val_loss=best_val_loss, best_val_epoch=best_val_epoch,
        )

        is_last_epoch = (epoch + 1) == epochs
        if save_every > 0 and ((epoch + 1) % save_every == 0 or is_last_epoch):
            ckpt_path = ckpt_dir / f"epoch_{epoch + 1:04d}.pt"
            save_checkpoint(
                ckpt_path, model, optimizer, scheduler, scaler,
                epoch + 1, global_step, raw, avg_loss,
                best_val_loss=best_val_loss, best_val_epoch=best_val_epoch,
            )
            print(f"saved checkpoint: {ckpt_path}")

        do_features = feature_every > 0 and ((epoch + 1) % feature_every == 0 or is_last_epoch)
        if do_features:
            try:
                feat = run_feature_tracking(
                    model, train_loader, test_loader, device, epoch + 1,
                    feature_csv_path, feature_max_samples,
                )
                if feat is not None:
                    metric = feat.get("knn_metric", "accuracy")
                    print(
                        f"  feature track ({metric}): knn1={feat['knn_k1']:.3f} "
                        f"knn5={feat['knn_k5']:.3f} "
                        f"eff_train={feat['eff_rank_train']:.2f} eff_test={feat['eff_rank_test']:.2f}"
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] feature tracking failed: {exc}")

        do_edges = decoder_type == "kan" and edge_every > 0 and (
            (epoch + 1) % edge_every == 0 or is_last_epoch
        )
        if do_edges:
            snap_path: Path | None = None
            if edge_snapshot_every > 0 and ((epoch + 1) % edge_snapshot_every == 0 or is_last_epoch):
                snap_path = snapshot_dir / f"epoch_{epoch + 1:04d}.npz"
            try:
                edge = run_edge_tracking(model.decoder, epoch + 1, edge_csv_path, snap_path)
                if edge is not None:
                    extra = f" | snapshot={snap_path.name}" if snap_path else ""
                    print(
                        f"  edge track: mean_tv={edge['edge_mean_tv']:.4f} "
                        f"max_tv={edge['edge_max_tv']:.4f} n_edges={edge['n_edges']}{extra}"
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] edge tracking failed: {exc}")

    total_train_seconds = time.time() - train_start_wall

    write_run_summary(
        output_dir, raw, decoder_type, epochs,
        last_train_loss, last_val_loss, best_val_loss, best_val_epoch,
        encoder_params, decoder_params, total_params, estimated_flops,
        device, total_train_seconds, seed, args.config, global_step,
    )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
