"""Shared setup for the command-line scripts (not meant to be run directly)."""
from __future__ import annotations

import os

# Must be set before torch / transformers are imported anywhere.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd

from edgevlm.config import ExperimentConfig
from edgevlm.constants import LABELS
from edgevlm.data.frame_paths import FrameResolver


def make_resolver(cfg: ExperimentConfig) -> FrameResolver:
    return FrameResolver(cfg.data_root, cfg.splits_dir, cfg.augmented_dir)


def read_split(cfg: ExperimentConfig, split: str) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv(split))
    required = {"frame_path", "label"}
    if not required <= set(df.columns):
        raise ValueError(f"{cfg.csv(split)} must contain columns {sorted(required)}")
    unknown = set(df["label"].astype(str).str.strip()) - set(LABELS)
    if unknown:
        raise ValueError(f"Unknown labels in {cfg.csv(split)}: {sorted(unknown)}")
    return df


def check_images_exist(df: pd.DataFrame, split: str, resolver: FrameResolver) -> None:
    missing = []
    for _, row in df.iterrows():
        try:
            resolver(row["frame_path"], split_name=split, label=row["label"])
        except FileNotFoundError:
            missing.append(row["frame_path"])
    print(f"{split} missing images: {len(missing)}")
    if missing:
        raise FileNotFoundError(f"{len(missing)} {split} images could not be resolved, e.g. {missing[:3]}")


def load_base_model(cfg: ExperimentConfig, device, dtype):
    """Local cache first, download only if the cache is incomplete."""
    from transformers import AutoModelForVision2Seq, AutoProcessor

    cache = str(cfg.model_cache)
    cfg.model_cache.mkdir(parents=True, exist_ok=True)
    try:
        processor = AutoProcessor.from_pretrained(cfg.model.model_id, cache_dir=cache, local_files_only=True)
        model = AutoModelForVision2Seq.from_pretrained(
            cfg.model.model_id, cache_dir=cache, local_files_only=True, torch_dtype=dtype
        )
    except OSError:
        print("Local cache incomplete; downloading model files...")
        processor = AutoProcessor.from_pretrained(cfg.model.model_id, cache_dir=cache)
        model = AutoModelForVision2Seq.from_pretrained(cfg.model.model_id, cache_dir=cache, torch_dtype=dtype)
    return processor, model.to(device)
