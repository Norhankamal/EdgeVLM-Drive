"""Experiment configuration (YAML -> typed dataclasses).

Every training run is described by one YAML file in ``configs/``. Relative paths under
``paths`` are resolved against ``paths.data_root`` (CSVs, frames) or ``paths.output_root``
(run outputs). ``$VARS`` / ``${VARS}`` in strings are expanded from the environment, so
machine-specific locations never need to be committed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from edgevlm.constants import IMAGE_SIZE, INSTRUCTIONS, MODEL_ID


@dataclass
class PathsConfig:
    data_root: str = "."          # base for frames and CSVs (frame_path columns are relative to it)
    splits_dir: str = "splits"
    augmented_dir: Optional[str] = None   # defaults to splits_dir
    train_csv: str = "train.csv"
    val_csv: str = "val.csv"
    test_csv: str = "test.csv"
    output_root: str = "runs"
    model_cache: str = "model_cache"


@dataclass
class DataConfig:
    preprocessing: str = "v2"     # "v1" (legacy pipeline) or "v2" (current pipeline)
    instruction: str = "v2"       # key of edgevlm.constants.INSTRUCTIONS


@dataclass
class ModelConfig:
    model_id: str = MODEL_ID
    image_size: int = IMAGE_SIZE


@dataclass
class LoraSettings:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05


@dataclass
class TrainingSettings:
    epochs: int = 8                      # upper bound; macro-F1 early stopping decides the real stop
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    logging_steps: int = 50
    save_strategy: str = "steps"         # "epoch" or "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    sampler_weight_power: float = 0.5    # 1.0 = full inverse frequency, 0.5 = square-root damping
    trainer_loads_best_by_eval_loss: bool = False
    warm_start_from_best: bool = True    # no usable checkpoint + earlier best adapter -> start from it


@dataclass
class ValidationCheckSettings:
    samples_per_class: int = 150
    patience: int = 3
    batch_size: int = 8
    seed: int = 0


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    description: str = ""
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraSettings = field(default_factory=LoraSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    validation_check: ValidationCheckSettings = field(default_factory=ValidationCheckSettings)

    # -- resolved locations ---------------------------------------------------------------
    @property
    def data_root(self) -> Path:
        return Path(self.paths.data_root)

    def _under_data_root(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.data_root / path

    @property
    def splits_dir(self) -> Path:
        return self._under_data_root(self.paths.splits_dir)

    @property
    def augmented_dir(self) -> Path:
        return self._under_data_root(self.paths.augmented_dir) if self.paths.augmented_dir else self.splits_dir

    def csv(self, split: str) -> Path:
        return self._under_data_root(getattr(self.paths, f"{'val' if split in ('val', 'validation') else split}_csv"))

    @property
    def output_dir(self) -> Path:
        base = Path(self.paths.output_root)
        return base / self.name

    @property
    def model_cache(self) -> Path:
        return Path(self.paths.model_cache)

    @property
    def instruction_text(self) -> str:
        return INSTRUCTIONS[self.data.instruction]


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def _build(cls, data: dict[str, Any]):
    """Recursively build dataclass ``cls`` from ``data``; unknown keys raise (typos must not pass silently)."""
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"Unknown key(s) {sorted(unknown)} for {cls.__name__}; valid: {sorted(known)}")
    defaults = cls()
    kwargs = {}
    for name, value in data.items():
        current = getattr(defaults, name)
        kwargs[name] = _build(type(current), value) if is_dataclass(current) and isinstance(value, dict) else value
    return cls(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = _build(ExperimentConfig, _expand(raw))
    if config.data.preprocessing not in ("v1", "v2"):
        raise ValueError(f"data.preprocessing must be 'v1' or 'v2', got {config.data.preprocessing!r}")
    if config.data.instruction not in INSTRUCTIONS:
        raise ValueError(f"data.instruction must be one of {sorted(INSTRUCTIONS)}, got {config.data.instruction!r}")
    if config.training.save_strategy not in ("epoch", "steps"):
        raise ValueError("training.save_strategy must be 'epoch' or 'steps'")
    return config
