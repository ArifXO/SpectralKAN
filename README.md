# SpectralKAN

Interpretable masked autoencoding for biosignals with a Kolmogorov-Arnold Network (KAN) decoder.

This project trains a Masked Autoencoder (MAE) on spectrogram-like biosignal inputs from ESC-50 audio and PTB-XL ECG data. The baseline uses a small Transformer decoder, while the main experimental model replaces the decoder with a KAN-based module so learned spline edges can be analyzed after pretraining.

## Project Layout

```text
configs/                  YAML experiment configs
data/                     Dataset loaders and local dataset folders
models/                   Encoder, decoders, and MAE wrapper
scripts/train.py          MAE pretraining entry point
scripts/evaluate.py       Linear probe and k-NN evaluation
scripts/freq_band_analysis.py
                           Frequency-band reconstruction comparison
notebooks/                Interactive experiment notebooks
results/                  Generated run folders, logs, checkpoints, and outputs
tools/                    Local tooling, such as FFmpeg
```

## Model

- Encoder: ViT-Small style encoder, 12 layers, 384 hidden dimension, 6 heads.
- Masking: 75 percent random patch masking.
- Baseline decoder: 2-layer Transformer decoder.
- KAN decoder: attention plus efficient-kan spline layers.
- Training loss: masked reconstruction MSE, with frequency-band metrics logged per epoch.

## Setup

Create and activate a Python environment, then install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The requirements file installs CUDA-enabled PyTorch wheels using the PyTorch CUDA index. Adjust the PyTorch install command if your machine uses a different CUDA/runtime setup.

## Data

Expected local dataset locations:

```text
data/ESC-50/
data/ptbxl/
```

ESC-50 should contain the dataset metadata and audio clips. PTB-XL should contain the PTB-XL records and metadata files expected by `data/ecg_dataset.py`.

## Training

Train the ESC-50 Transformer baseline:

```powershell
python scripts\train.py --config configs\esc50_config.yaml
```

Train the ESC-50 KAN decoder:

```powershell
python scripts\train.py --config configs\esc50_kan_config.yaml
```

Train PTB-XL variants:

```powershell
python scripts\train.py --config configs\ptbxl_config.yaml
python scripts\train.py --config configs\ptbxl_kan_config.yaml
```

Useful overrides:

```powershell
python scripts\train.py --config configs\esc50_config.yaml --epochs 5 --batch-size 32 --num-workers 0
python scripts\train.py --config configs\esc50_config.yaml --resume results\<run>\checkpoints\last.pt
```

Each fresh training run creates a folder under `results/`:

```text
results/<timestamp>_<dataset>_<decoder>/
  config.yaml
  training_log.csv
  training_result.yaml
  checkpoints/
```

On Windows, `--num-workers 0` can be useful if multiprocessing DataLoader workers fail.

## Evaluation

Run linear probe and k-NN evaluation from a checkpoint:

```powershell
python scripts\evaluate.py --checkpoint results\<run>\checkpoints\last.pt --dataset esc50
```

By default, evaluation results are written back into the checkpoint's run folder:

```text
results/<run>/evaluation/eval_results.csv
```

## Frequency-Band Analysis

Compare Transformer and KAN checkpoints:

```powershell
python scripts\freq_band_analysis.py `
  --transformer-checkpoint results\<transformer-run>\checkpoints\last.pt `
  --kan-checkpoint results\<kan-run>\checkpoints\last.pt `
  --dataset esc50
```

Outputs are written to the common checkpoint run folder when possible, or to a timestamped folder under `results/freq_band/`.

## Notes

- Do not use ViT-Large on the target GPUs.
- Keep batch size at or below 128 on the RTX 3060 and 256 on the RTX 4070 Ti Super.
- Do not change the 0.75 masking ratio without updating the relevant config.
- Large local artifacts, datasets, checkpoints, and generated results are intentionally ignored by Git.
