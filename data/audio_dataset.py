from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FFMPEG_BIN = _PROJECT_ROOT / "tools" / "ffmpeg" / "bin"
_FFMPEG_DLL_DIR = None
if _FFMPEG_BIN.exists():
    os.environ["PATH"] = f"{_FFMPEG_BIN}{os.pathsep}{os.environ.get('PATH', '')}"
    if hasattr(os, "add_dll_directory"):
        _FFMPEG_DLL_DIR = os.add_dll_directory(str(_FFMPEG_BIN))

import torchaudio


class ESC50Dataset(Dataset):
    """ESC-50 audio dataset returning normalized mel spectrograms."""

    SAMPLE_RATE = 22050
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 512
    CLIP_SECONDS = 5
    NUM_SAMPLES = SAMPLE_RATE * CLIP_SECONDS

    def __init__(
        self,
        root: str | Path = "data/ESC-50",
        split: str = "train",
        val_fold: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.audio_dir = self.root / "audio"
        self.metadata_path = self.root / "meta" / "esc50.csv"
        self.split = split
        self.val_fold = val_fold

        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")
        if val_fold is not None and val_fold not in {1, 2, 3, 4}:
            raise ValueError("val_fold must be one of 1, 2, 3, 4 when provided")
        if split == "val" and val_fold is None:
            raise ValueError("val split requires val_fold to be set")
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"ESC-50 metadata not found at {self.metadata_path}. "
                "Expected the standard ESC-50 layout with meta/esc50.csv."
            )
        if not self.audio_dir.exists():
            raise FileNotFoundError(
                f"ESC-50 audio directory not found at {self.audio_dir}."
            )

        self.samples = self._load_samples()
        if not self.samples:
            raise ValueError(
                f"No ESC-50 samples found for split={split!r}, val_fold={val_fold!r}."
            )

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.SAMPLE_RATE,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            n_mels=self.N_MELS,
        )
        self.resamplers: dict[int, torchaudio.transforms.Resample] = {}

    def _load_samples(self) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []

        with self.metadata_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            required_columns = {"filename", "fold", "target"}
            missing_columns = required_columns.difference(reader.fieldnames or [])
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(f"ESC-50 metadata is missing columns: {missing}")

            for row in reader:
                fold = int(row["fold"])
                if not self._uses_fold(fold):
                    continue

                audio_path = self.audio_dir / row["filename"]
                if not audio_path.exists():
                    raise FileNotFoundError(f"Audio file listed in metadata is missing: {audio_path}")

                samples.append((audio_path, int(row["target"])))

        return samples

    def _uses_fold(self, fold: int) -> bool:
        if self.split == "test":
            return fold == 5
        if self.split == "val":
            return fold == self.val_fold
        if self.val_fold is None:
            return fold in {1, 2, 3, 4}
        return fold in {1, 2, 3, 4} and fold != self.val_fold

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        audio_path, label = self.samples[index]
        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != self.SAMPLE_RATE:
            if sample_rate not in self.resamplers:
                self.resamplers[sample_rate] = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.SAMPLE_RATE,
                )
            waveform = self.resamplers[sample_rate](waveform)

        if waveform.size(1) < self.NUM_SAMPLES:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.NUM_SAMPLES - waveform.size(1))
            )
        elif waveform.size(1) > self.NUM_SAMPLES:
            waveform = waveform[:, : self.NUM_SAMPLES]

        spectrogram = self.mel_transform(waveform)
        spectrogram = torch.log1p(spectrogram)
        spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-6)

        return spectrogram, label


def get_esc50_loaders(
    batch_size: int,
    num_workers: int,
    root: str | Path = "data/ESC-50",
    val_fold: Optional[int] = None,
) -> tuple[DataLoader, Optional[DataLoader], DataLoader]:
    train_dataset = ESC50Dataset(root=root, split="train", val_fold=val_fold)
    val_dataset = (
        ESC50Dataset(root=root, split="val", val_fold=val_fold)
        if val_fold is not None
        else None
    )
    test_dataset = ESC50Dataset(root=root, split="test")

    pin_memory = torch.cuda.is_available()
    persistent = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
        )
        if val_dataset is not None
        else None
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    return train_loader, val_loader, test_loader
