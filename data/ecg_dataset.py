from __future__ import annotations

import ast
import warnings
from pathlib import Path

import pandas as pd
import torch
import wfdb
from torch.utils.data import DataLoader, Dataset


class PTBXLDataset(Dataset):
    """PTB-XL ECG dataset returning normalized per-lead STFT spectrograms."""

    SAMPLE_RATE = 100
    NUM_LEADS = 12
    NUM_SAMPLES = 1000
    SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
    N_FFT = 256
    HOP_LENGTH = 10
    WIN_LENGTH = 256
    TARGET_FREQ = 128
    TARGET_TIME = 128

    def __init__(
        self,
        root: str | Path = "data/ptbxl",
        split: str = "train",
        download: bool = False,
    ) -> None:
        self.root = Path(root)
        self.metadata_path = self.root / "ptbxl_database.csv"
        self.scp_path = self.root / "scp_statements.csv"
        self.split = split

        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        if download and (not self.metadata_path.exists() or not self.scp_path.exists()):
            self._download_ptbxl()

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"PTB-XL metadata not found at {self.metadata_path}. "
                "Set download=True or place the official PTB-XL files under data/ptbxl/."
            )
        if not self.scp_path.exists():
            raise FileNotFoundError(
                f"PTB-XL SCP statements not found at {self.scp_path}."
            )

        self.code_to_superclass = self._load_code_to_superclass()
        self.samples = self._load_samples(download=download)
        if not self.samples:
            raise ValueError(f"No PTB-XL samples found for split={split!r}.")

        self.window = torch.hann_window(self.WIN_LENGTH)

    def _download_ptbxl(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        metadata_files = [
            file_name
            for file_name in ("ptbxl_database.csv", "scp_statements.csv")
            if not (self.root / file_name).exists()
        ]
        if metadata_files:
            wfdb.dl_files(
                "ptb-xl",
                dl_dir=str(self.root),
                files=metadata_files,
                keep_subdirs=True,
                overwrite=False,
            )

    def _download_records(self, missing_records: list[Path]) -> None:
        files = []
        for record_path in missing_records:
            rel = record_path.relative_to(self.root)
            for suffix in (".hea", ".dat"):
                files.append(rel.with_suffix(suffix).as_posix())
        if files:
            wfdb.dl_files(
                "ptb-xl",
                dl_dir=str(self.root),
                files=files,
                keep_subdirs=True,
                overwrite=False,
            )

    def _load_code_to_superclass(self) -> dict[str, str]:
        scp = self._read_scp_statements()
        required_columns = {"diagnostic", "diagnostic_class"}
        missing_columns = required_columns.difference(scp.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"scp_statements.csv is missing columns: {missing}")

        code_to_superclass: dict[str, str] = {}
        diagnostic_rows = scp[scp["diagnostic"] == 1]
        for code, row in diagnostic_rows.iterrows():
            superclass = row["diagnostic_class"]
            if superclass in self.SUPERCLASSES:
                code_to_superclass[str(code)] = str(superclass)

        return code_to_superclass

    def _read_scp_statements(self) -> pd.DataFrame:
        scp = pd.read_csv(self.scp_path, index_col=0, skipinitialspace=True)
        scp.index = scp.index.astype(str).str.strip()
        scp.index.name = "code"
        for col in scp.select_dtypes(include="object").columns:
            scp[col] = scp[col].astype(str).str.strip()
        return scp

    def _load_samples(self, download: bool = False) -> list[tuple[Path, torch.Tensor]]:
        metadata = pd.read_csv(self.metadata_path)
        required_columns = {"filename_lr", "scp_codes", "strat_fold"}
        missing_columns = required_columns.difference(metadata.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"ptbxl_database.csv is missing columns: {missing}")

        candidates: list[tuple[Path, torch.Tensor]] = []
        missing_records: list[Path] = []
        for row in metadata.itertuples(index=False):
            fold = int(getattr(row, "strat_fold"))
            if not self._uses_fold(fold):
                continue

            record_path = self.root / getattr(row, "filename_lr")
            label = self._make_label(getattr(row, "scp_codes"))
            if record_path.with_suffix(".hea").exists() and record_path.with_suffix(".dat").exists():
                candidates.append((record_path, label))
            else:
                missing_records.append(record_path)
                if download:
                    candidates.append((record_path, label))

        if missing_records and download:
            self._download_records(missing_records)
        elif missing_records:
            warnings.warn(
                f"Skipped {len(missing_records)} PTB-XL records missing local .hea/.dat files "
                f"for split={self.split!r}. Pass download=True to fetch them.",
                RuntimeWarning,
                stacklevel=2,
            )

        return candidates

    def _uses_fold(self, fold: int) -> bool:
        if self.split == "test":
            return fold == 10
        if self.split == "val":
            return fold == 9
        return 1 <= fold <= 8

    def _make_label(self, scp_codes: str) -> torch.Tensor:
        labels = torch.zeros(len(self.SUPERCLASSES), dtype=torch.float32)
        codes = ast.literal_eval(scp_codes)
        for code in codes:
            superclass = self.code_to_superclass.get(str(code))
            if superclass is not None:
                labels[self.SUPERCLASSES.index(superclass)] = 1.0
        return labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record_path, label = self.samples[index]
        record = wfdb.rdrecord(str(record_path), physical=True)
        signal = torch.as_tensor(record.p_signal, dtype=torch.float32).transpose(0, 1)

        if signal.size(0) != self.NUM_LEADS:
            raise ValueError(f"Expected {self.NUM_LEADS} ECG leads, got {signal.size(0)} for {record_path}")

        if signal.size(1) < self.NUM_SAMPLES:
            signal = torch.nn.functional.pad(signal, (0, self.NUM_SAMPLES - signal.size(1)))
        elif signal.size(1) > self.NUM_SAMPLES:
            signal = signal[:, : self.NUM_SAMPLES]

        signal = torch.nan_to_num(signal)
        stft = torch.stft(
            signal,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            win_length=self.WIN_LENGTH,
            window=self.window,
            return_complex=True,
        )
        spectrogram = stft.abs()

        F = spectrogram.size(1)
        if F >= self.TARGET_FREQ:
            spectrogram = spectrogram[:, : self.TARGET_FREQ, :]
        else:
            spectrogram = torch.nn.functional.pad(spectrogram, (0, 0, 0, self.TARGET_FREQ - F))

        T = spectrogram.size(2)
        if T >= self.TARGET_TIME:
            spectrogram = spectrogram[:, :, : self.TARGET_TIME]
        else:
            spectrogram = torch.nn.functional.pad(spectrogram, (0, self.TARGET_TIME - T))

        mean = spectrogram.mean(dim=(1, 2), keepdim=True)
        std = spectrogram.std(dim=(1, 2), keepdim=True)
        spectrogram = (spectrogram - mean) / (std + 1e-6)

        return spectrogram, label


def get_ptbxl_loaders(
    batch_size: int,
    num_workers: int,
    root: str | Path = "data/ptbxl",
    download: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Backward-compatible 3-way loader factory.

    Standard PTB-XL stratified split: ``strat_fold`` 1-8 = train,
    fold 9 = val, fold 10 = test.
    """
    train_dataset = PTBXLDataset(root=root, split="train", download=download)
    val_dataset = PTBXLDataset(root=root, split="val", download=download)
    test_dataset = PTBXLDataset(root=root, split="test", download=download)

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
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
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
