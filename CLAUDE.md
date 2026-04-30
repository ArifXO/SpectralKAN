# CLAUDE.md — Project Context for Claude Code

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
- models/mae.py: full MAE framework
- data/ecg_dataset.py: PTB-XL loader
- data/audio_dataset.py: ESC-50 loader
- train.py: main training script
- evaluate.py: linear probe evaluation
- analyze_edges.py: KAN edge visualization

## DO NOT
- Do not use ViT-Large (too big for our GPU)
- Do not use batch size > 128 on RTX 3060
- Do not use batch size > 256 on RTX 4070 Ti Super
- Do not change the masking ratio from 0.75 without updating the config
