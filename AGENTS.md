# AGENTS.md - Project Context

## Project Name
SpectralKAN: Interpretable Masked Autoencoding for Biosignals via KAN Decoder

## What We Are Building
A Masked Autoencoder (MAE) where the decoder is a Kolmogorov-Arnold Network (KAN).
We pretrain on biosignals (ECG from PTB-XL, audio from ESC-50).
After training, we analyze the learned KAN spline edges for interpretability.

## Architecture
- Encoder: ViT-Small (12 layers, 384 hidden dim, 6 heads)
- Decoder (baseline): Transformer decoder, 2 blocks
- Decoder (ours): KAN-based decoder using efficient-kan library
- Masking: 75% of patches masked during pretraining

## Hardware
- Personal PC: RTX 3060 12GB, use for small experiments
- University PC: RTX 4070 Ti Super 16GB, use for full runs

## Key Libraries
- PyTorch 2.x with CUDA
- timm (for ViT encoder)
- einops (for tensor reshaping)
- efficient-kan (for KAN layers)
- torchaudio (for audio processing)
- wfdb (for ECG/PTB-XL loading)
- scipy, matplotlib, wandb

## Datasets
- ESC-50: /data/ESC-50/ (2000 audio clips, 50 classes)
- PTB-XL: /data/ptbxl/ (21837 ECG records, 12 leads)

## Key File Locations
- models/encoder.py: ViT encoder
- models/decoder_transformer.py: baseline decoder
- models/decoder_kan.py: KAN decoder (our contribution)
- models/decoder_utils.py: shared mask-token splice helper
- models/mae.py: full MAE framework
- data/ecg_dataset.py: PTB-XL loader
- data/audio_dataset.py: ESC-50 loader
- scripts/train.py: main training script
- scripts/evaluate.py: linear probe evaluation
- scripts/freq_band_analysis.py: frequency-band reconstruction analysis
- utils/setup.py: config loading, dataloaders, optimizer/scheduler, seeds, run dirs
- utils/training_loop.py: train_one_epoch / validate_one_epoch / band-loss helpers
- utils/periodic_evals.py: periodic feature-tracking + KAN edge-tracking helpers
- utils/checkpointing.py: checkpoint save/load and run summary
- utils/logging_utils.py: CSV + wandb logging helpers
- utils/metrics.py: pure metric helpers (band MSE, kNN, FLOPs, GPU mem)
- utils/edge_tracker.py: KAN edge statistics and snapshot helpers
- utils/output_paths.py: run-dir resolution helpers

## Results Layout
- results/sweeps/: 50-epoch hyperparameter-search runs and sweep summary CSVs
- results/runs/: full or main training runs
- Fresh training supports readable run folders:
  - --run-name <name>: use a readable folder stem instead of a timestamp
  - --output-dir <path>: override logging.output_dir from the config
- Use results/sweeps for HP search commands, for example:
  - python scripts/train.py --config configs/esc50_kan_config.yaml --epochs 50 --run-name esc50_kan_lr1e-4_grid5_seed42 --output-dir results/sweeps
- Use results/runs for full training commands, for example:
  - python scripts/train.py --config configs/esc50_kan_config.yaml --epochs 200 --run-name esc50_kan_full_seed42 --output-dir results/runs
- If --run-name collides with an existing folder, the code appends _1, _2, etc.
- --run-dir remains the exact-directory override and takes precedence over --run-name.

## DO NOT
- Do not use ViT-Large (too big for our GPU)
- Do not use batch size > 128 on RTX 3060
- Do not use batch size > 256 on RTX 4070 Ti Super
- Do not change the masking ratio from 0.75 without updating the config

## Sync Setup
- Syncthing is running on both machines and syncs these folders automatically:
  - results/ -> checkpoints, CSV logs, figures, eval results
  - training_logs/ -> nohup output logs
- Code changes go through Git (git push / git pull)
- Checkpoint/result files come through Syncthing (no Git needed for those)
- Personal PC path: A:\UNI\CSE400\SpectralKAN
- University PC path: E:\CSE400\SpectralKAN
