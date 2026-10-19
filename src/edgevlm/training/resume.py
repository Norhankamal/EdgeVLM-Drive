"""Resume-from-checkpoint helpers for multi-day runs on a shared server.

A process killed mid-write can leave a ``checkpoint-<step>`` folder that exists but is
truncated, so a checkpoint is only used if every file ``Trainer`` needs is present *and*
readable; otherwise the previous checkpoint is tried.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

CHECKPOINT_REQUIRED_FILES = (
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "adapter_model.safetensors",
)


def allow_trusted_torch_load() -> bool:
    """Make ``torch.load`` default to ``weights_only=False`` on torch >= 2.6.

    ``Trainer.train(resume_from_checkpoint=...)`` reads ``rng_state.pth`` with a plain
    ``torch.load``; from torch 2.6 the default ``weights_only=True`` rejects the numpy RNG
    state stored there and resuming fails. Only call this for checkpoints written by this
    project's own runs. Returns True if the patch was applied.
    """
    import torch

    major_minor = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    if major_minor < (2, 6):
        return False
    original = torch.load

    def _trusted_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = _trusted_torch_load
    return True


def checkpoint_is_usable(checkpoint_dir: Path) -> bool:
    """True only if every required file exists and is readable."""
    checkpoint_dir = Path(checkpoint_dir)
    if not all((checkpoint_dir / name).is_file() for name in CHECKPOINT_REQUIRED_FILES):
        return False
    try:
        import torch
        from safetensors import safe_open

        json.loads((checkpoint_dir / "trainer_state.json").read_text())
        with safe_open(str(checkpoint_dir / "adapter_model.safetensors"), framework="pt") as handle:
            list(handle.keys())
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
            torch.load(str(checkpoint_dir / name), map_location="cpu")
    except Exception:
        return False
    return True


def find_resume_checkpoint(output_dir: Path, usable=checkpoint_is_usable) -> Optional[Path]:
    """Newest ``checkpoint-<step>`` folder under ``output_dir`` that is usable, else None."""
    candidates = []
    for path in Path(output_dir).glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match and path.is_dir():
            candidates.append((int(match.group(1)), path))
    for step, path in sorted(candidates, reverse=True):
        if usable(path):
            return path
        print(f"Skipping checkpoint-{step}: incomplete or unreadable (interrupted while being written).")
    return None
