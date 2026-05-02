# SpectralKAN

Interpretable masked autoencoding for biosignals with a Kolmogorov-Arnold Network (KAN) decoder.

SpectralKAN trains a Masked Autoencoder (MAE) on time-frequency biosignal inputs, then compares a standard Transformer decoder against a KAN-based decoder. The research goal is not only reconstruction quality, but also whether the learned KAN spline edges expose useful structure in frequency bands and biosignal representations.

## What This Repo Contains

- MAE pretraining for ESC-50 audio and PTB-XL ECG spectrograms.
- A ViT-Small style spectral encoder shared by all experiments.
- Two decoder families:
  - Transformer decoder baseline.
  - KAN decoder using `efficient-kan`, with sampled spline edges for interpretability.
- Training, checkpointing, feature tracking, KAN edge tracking, linear probe evaluation, k-NN evaluation, and frequency-band reconstruction analysis.

## Architecture

The default experiment setup is defined in `configs/base_config.yaml`.

| Component | Default |
|---|---|
| Encoder | ViT-Small style spectral encoder |
| Encoder depth | 12 Transformer blocks |
| Encoder width | 384 |
| Encoder heads | 6 |
| Patch size | 16 |
| Masking ratio | 0.75 |
| Baseline decoder | 2-block Transformer decoder |
| KAN decoder | Self-attention mixer + 2 `KANLinear` layers |
| Decoder width | 512 |
| KAN hidden dim | 512 |
| KAN grid size | 5 |
| KAN spline order | 3 |
| Loss | Masked-patch MSE with normalized pixel targets |

High-level flow:

```text
spectrogram input
  -> patchify and random 75% masking
  -> SpectralViTEncoder
  -> TransformerDecoder or KANDecoder
  -> masked patch reconstruction loss
  -> optional feature, band-loss, and KAN edge analysis
```

The KAN decoder has the same input/output contract as the Transformer decoder, so experiments can switch decoders through config or CLI overrides.

## Repository Layout

```text
configs/
  base_config.yaml          Shared model, training, and logging defaults
  esc50_config.yaml         ESC-50 Transformer baseline
  esc50_kan_config.yaml     ESC-50 KAN decoder experiment
  ptbxl_config.yaml         PTB-XL Transformer baseline
  ptbxl_kan_config.yaml     PTB-XL KAN decoder experiment

data/
  audio_dataset.py          ESC-50 loader and mel-spectrogram transform
  ecg_dataset.py            PTB-XL loader and per-lead STFT transform

models/
  encoder.py                Spectral ViT encoder with MAE masking
  decoder_transformer.py    Transformer decoder baseline
  decoder_kan.py            KAN decoder and spline extraction methods
  decoder_utils.py          Shared mask-token splice helper
  mae.py                    Full MAE wrapper and reconstruction utilities

scripts/
  train.py                  MAE pretraining entry point
  evaluate.py               Linear probe and k-NN evaluation
  freq_band_analysis.py     Per-band reconstruction comparison

utils/
  setup.py                  Config loading, loaders, optimizer, scheduler, seeds
  training_loop.py          Train/validation loops and band-loss helpers
  periodic_evals.py         Periodic feature and KAN edge tracking
  checkpointing.py          Checkpoint save/load and run summaries
  logging_utils.py          CSV and wandb logging
  metrics.py                Band MSE, k-NN, FLOPs, GPU memory helpers
  edge_tracker.py           KAN edge statistics and snapshots
  output_paths.py           Run directory helpers

docs/
  syncthing.md              Local artifact sync notes for the two-PC setup
```

## Environment Setup

Create a Python environment and install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` uses the PyTorch CUDA wheel index and installs `efficient-kan` from GitHub. If your CUDA/runtime setup differs, install the appropriate PyTorch build first, then reinstall the remaining requirements.

Core dependencies include PyTorch, torchaudio, timm, einops, efficient-kan, wfdb, scipy, scikit-learn, matplotlib, seaborn, wandb, and PyYAML.

## Data Layout

Expected dataset roots:

```text
data/ESC-50/
data/ptbxl/
```

### ESC-50

Expected standard ESC-50 structure:

```text
data/ESC-50/
  audio/
  meta/esc50.csv
```

The loader converts each clip to mono, resamples to 22050 Hz when needed, pads/truncates to 5 seconds, then returns a normalized log mel spectrogram.

Default split:

| Split | Folds |
|---|---|
| Train | 1, 2, 3 |
| Validation | 4 |
| Test | 5 |

Pass `data.val_fold: null` or call the loader with `val_fold=None` to use folds 1-4 for training and no validation split.

### PTB-XL

Expected PTB-XL files include:

```text
data/ptbxl/
  ptbxl_database.csv
  scp_statements.csv
  records100/
```

The loader reads the low-resolution 100 Hz records, creates 12-lead STFT magnitude spectrograms, crops/pads them to `12 x 128 x 128`, and produces multi-label superclass targets:

```text
NORM, MI, STTC, CD, HYP
```

Default split follows PTB-XL stratified folds:

| Split | Stratified folds |
|---|---|
| Train | 1-8 |
| Validation | 9 |
| Test | 10 |

`data.download: true` can be added to a PTB-XL config to let `wfdb` fetch missing metadata and records.

## Training

Run the Transformer baseline on ESC-50:

```powershell
python scripts\train.py --config configs\esc50_config.yaml
```

Run the KAN decoder on ESC-50:

```powershell
python scripts\train.py --config configs\esc50_kan_config.yaml
```

Run PTB-XL variants:

```powershell
python scripts\train.py --config configs\ptbxl_config.yaml
python scripts\train.py --config configs\ptbxl_kan_config.yaml
```

Useful CLI overrides:

```powershell
python scripts\train.py --config configs\esc50_config.yaml --epochs 5 --batch-size 32 --num-workers 0
python scripts\train.py --config configs\esc50_config.yaml --decoder-type kan --decoder-lr 5.0e-5
python scripts\train.py --config configs\esc50_config.yaml --device cpu
```

Resume from a checkpoint:

```powershell
python scripts\train.py --config configs\esc50_config.yaml --resume results\<run>\checkpoints\last.pt
```

Use an explicit run directory:

```powershell
python scripts\train.py --config configs\esc50_kan_config.yaml --run-dir results\esc50_kan_debug
```

Main training arguments:

| Argument | Purpose |
|---|---|
| `--config` | Required YAML config path |
| `--resume` | Resume optimizer, scheduler, scaler, epoch, and model state |
| `--epochs` | Override `training.epochs` |
| `--batch-size` | Override `training.batch_size` |
| `--save-every` | Override `logging.save_every` |
| `--device` | Override `training.device` |
| `--num-workers` | Override `data.num_workers` |
| `--seed` | Override `training.seed` |
| `--decoder-type` | Override decoder type: `transformer` or `kan` |
| `--decoder-lr` | Use a separate decoder learning rate |
| `--run-dir` | Write into a specific output directory |

On Windows, `--num-workers 0` is often the simplest option when DataLoader multiprocessing causes issues.

## Training Outputs

Fresh runs create timestamped folders under `results/`:

```text
results/<timestamp>_<dataset>_<decoder>/
  config.yaml
  training_log.csv
  run_summary.txt
  training_result.yaml
  feature_tracking.csv
  edge_evolution.csv          # KAN runs only, when edge tracking is enabled
  checkpoints/
    last.pt
    best_model.pt
    epoch_0020.pt
    ...
  edge_snapshots/
    epoch_0050.npz            # KAN runs only, when snapshotting is enabled
```

`training_log.csv` includes train loss, validation loss, low/mid/high band losses, gradient norms, learning rate, epoch time, and peak GPU memory. KAN edge tracking records mean and max total variation across sampled spline edges.

Default logging cadence:

| Config key | Default | Meaning |
|---|---:|---|
| `logging.save_every` | 20 | Save numbered checkpoints every N epochs |
| `logging.feature_every` | 25 | Run feature tracking every N epochs |
| `logging.edge_every` | 25 | Log KAN edge statistics every N epochs |
| `logging.edge_snapshot_every` | 50 | Save raw KAN spline snapshots every N epochs |

`feature_every`, `edge_every`, and `edge_snapshot_every` are optional config keys. If absent, the defaults above are used by `scripts/train.py`.

## Evaluation

Evaluate a pretrained checkpoint with a frozen encoder:

```powershell
python scripts\evaluate.py --checkpoint results\<run>\checkpoints\last.pt --dataset esc50
python scripts\evaluate.py --checkpoint results\<run>\checkpoints\last.pt --dataset ptbxl
```

Evaluation extracts CLS-token features once for train/test splits, then runs:

- Linear probe with a single `nn.Linear` head.
- Cosine-similarity k-NN for `k=1` and `k=5` by default.

Metrics:

| Dataset | Task | Metric |
|---|---|---|
| ESC-50 | Multiclass, 50 classes | Accuracy |
| PTB-XL | Multilabel, 5 superclasses | Macro AUROC |

Common options:

```powershell
python scripts\evaluate.py `
  --checkpoint results\<run>\checkpoints\best_model.pt `
  --dataset esc50 `
  --linear-epochs 50 `
  --batch-size 128 `
  --num-workers 0
```

By default, results are appended to:

```text
results/<run>/evaluation/eval_results.csv
```

Use `--output path\to\eval_results.csv` to write somewhere else.

## Frequency-Band Reconstruction Analysis

Compare a Transformer checkpoint against a KAN checkpoint:

```powershell
python scripts\freq_band_analysis.py `
  --transformer-checkpoint results\<transformer-run>\checkpoints\last.pt `
  --kan-checkpoint results\<kan-run>\checkpoints\last.pt `
  --dataset esc50
```

For PTB-XL:

```powershell
python scripts\freq_band_analysis.py `
  --transformer-checkpoint results\<transformer-run>\checkpoints\last.pt `
  --kan-checkpoint results\<kan-run>\checkpoints\last.pt `
  --dataset ptbxl
```

Outputs:

```text
freq_band_mse_<dataset>.csv
freq_band_mse_<dataset>_equal.png
freq_band_mse_<dataset>_clinical.png   # PTB-XL only
```

The analysis always reports equal thirds along the frequency axis. For PTB-XL it also reports clinical ECG bands: 0-5 Hz, 5-15 Hz, and 15-40 Hz.

## KAN Interpretability Hooks

The KAN decoder exposes two main methods:

```python
decoder.get_edge_functions(num_points=200)
decoder.get_spline_coefficients()
```

`utils.edge_tracker` builds on these methods to:

- Sample every KAN edge function.
- Compute mean and max total variation across spline edges.
- Save raw spline parameters to `.npz` snapshots for later plotting or pruning analysis.

Transformer decoders safely return no KAN edge stats, so the same tracking code can run across both decoder families.

## Config Notes

Configs support inheritance through `inherits:`. Child configs are deep-merged over their parent, so `configs/esc50_kan_config.yaml` only needs to override the decoder and dataset-specific fields from `base_config.yaml`.

Important defaults:

```yaml
training:
  epochs: 200
  batch_size: 128
  lr: 1.5e-4
  warmup_epochs: 40
  weight_decay: 0.05
  device: cuda
  mixed_precision: true
  seed: 42

logging:
  use_wandb: false
  save_every: 20
  output_dir: results/
```

For KAN ESC-50, the config sets a smaller decoder learning rate:

```yaml
training:
  decoder_lr: 5.0e-5
```

To enable Weights & Biases, add or override:

```yaml
logging:
  use_wandb: true
  wandb_project: spectral-kan
```

## Project Constraints

The training setup enforces the main project guardrails before expensive work starts:

- Only `vit_small` is supported.
- Masking ratio must remain `0.75`.
- Batch size must stay within the target GPU limits:
  - RTX 3060 12GB: maximum 128.
  - RTX 4070 Ti Super 16GB: maximum 256.

If you intentionally change any of these assumptions, update the relevant config and validation logic together.

## Two-PC Workflow

Source code changes go through Git. Runtime artifacts are synced separately.

| Machine | Path |
|---|---|
| Personal PC | `A:\UNI\CSE400\SpectralKAN` |
| University PC | `E:\CSE400\SpectralKAN` |

Syncthing handles:

```text
results/
training_logs/
```

That means checkpoints, CSV logs, figures, and long-run console logs can move between machines without committing them. See `docs/syncthing.md` for details.

## Development Checks

Run lightweight module smoke tests:

```powershell
python models\encoder.py
python models\decoder_transformer.py
python models\decoder_kan.py
python models\mae.py
python utils\edge_tracker.py
```

Run pytest if tests are added or available:

```powershell
pytest
```

## Troubleshooting

CUDA requested but unavailable:

```text
[warn] CUDA requested but not available; falling back to CPU.
```

Install a CUDA-enabled PyTorch build compatible with the local driver, or run with `--device cpu` for smoke tests.

DataLoader hangs or crashes on Windows:

```powershell
python scripts\train.py --config configs\esc50_config.yaml --num-workers 0
```

Missing ESC-50 files:

```text
Expected data/ESC-50/meta/esc50.csv and data/ESC-50/audio/
```

Missing PTB-XL files:

```text
Expected data/ptbxl/ptbxl_database.csv, scp_statements.csv, and record files.
```

For PTB-XL, set `data.download: true` in the config if you want `wfdb` to fetch missing official files.

Resume checkpoint fails architecture validation:

```text
Resume checkpoint architecture does not match the active config.
```

Use the same config and decoder settings that produced the checkpoint, or start a fresh run.
