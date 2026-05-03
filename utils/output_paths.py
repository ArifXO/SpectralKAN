from __future__ import annotations

from datetime import datetime
from pathlib import Path


def make_run_dir(
    base_output_dir: str | Path,
    dataset: str,
    decoder: str,
    run_name: str | None = None,
) -> Path:
    """Create a unique run directory under ``base_output_dir``.

    If ``run_name`` is provided, the directory stem is exactly that name.
    Otherwise, the legacy ``<timestamp>_<dataset>_<decoder>`` stem is used.
    Existing stems are made unique by appending ``_1``, ``_2``, and so on.
    """
    base = Path(base_output_dir)
    if run_name is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{stamp}_{dataset}_{decoder}"
    else:
        stem = run_name
    run_dir = base / stem
    suffix = 1
    while run_dir.exists():
        run_dir = base / f"{stem}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def infer_run_dir_from_checkpoint(checkpoint: str | Path) -> Path:
    """Return the run directory that owns a checkpoint path.

    Expected project layout is:
        results/<run_name>/checkpoints/<checkpoint>.pt

    If a checkpoint is not inside a ``checkpoints`` folder, fall back to the
    checkpoint's parent directory so ad-hoc checkpoints still get local outputs.
    """
    ckpt = Path(checkpoint)
    parent = ckpt.parent
    if parent.name == "checkpoints":
        return parent.parent
    return parent


def infer_common_run_dir(checkpoints: list[str | Path]) -> Path | None:
    """Return a shared run directory when all checkpoints belong to one run."""
    run_dirs = [infer_run_dir_from_checkpoint(path).resolve() for path in checkpoints]
    first = run_dirs[0]
    if all(path == first for path in run_dirs):
        return first
    return None
