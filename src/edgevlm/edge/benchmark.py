"""Latency and memory measurement for the deployed model on the target device."""

from __future__ import annotations

import time
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from edgevlm.constants import LABELS


def balanced_subset(df: pd.DataFrame, per_class: int, label_col: str = "label") -> pd.DataFrame:
    """``per_class`` rows per label, taken at even spacing through each class (deterministic, no random seed)."""
    parts = []
    for label in LABELS:
        rows = df[df[label_col].astype(str).str.strip() == label]
        if len(rows) < per_class:
            raise ValueError(f"only {len(rows)} {label} rows, need {per_class}")
        idx = np.linspace(0, len(rows) - 1, per_class).round().astype(int)
        parts.append(rows.iloc[idx])
    return pd.concat(parts).reset_index(drop=True)


def summarise_latency(latencies_ms: Sequence[float]) -> dict:
    lat = np.asarray(latencies_ms, dtype=float)
    mean = float(lat.mean())
    return {
        "mean_ms": mean,
        "p95_ms": float(np.percentile(lat, 95)),
        "throughput_frames_per_min": 60000.0 / mean,
        "n": int(len(lat)),
    }


def peak_gpu_memory_mb() -> Optional[float]:
    """Peak GPU memory allocated by PyTorch since the last reset, or ``None`` without CUDA."""
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1024**2


def time_calls(fn: Callable[[], object], n_warmup: int = 2, n: int = 10) -> list[float]:
    """Wall-clock milliseconds for ``n`` calls of ``fn`` after ``n_warmup`` untimed calls."""
    for _ in range(n_warmup):
        fn()
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000)
    return out
