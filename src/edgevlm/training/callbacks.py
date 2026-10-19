"""Training callbacks.

``MacroF1EarlyStopping`` is the checkpoint-selection mechanism of this project. With a
strongly imbalanced dataset, ``eval_loss`` can look good while recall on the rare,
safety-critical class (DROWSY) collapses, so after every epoch this callback runs real
``generate()`` calls on a fixed, class-balanced validation subset, computes macro-F1, saves
the adapter whenever it improves (``best_by_macro_f1/``) and stops after ``patience``
consecutive epochs without improvement.

The per-class numbers it logs (``per_class_f1``) are per-class **F1** scores.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

try:
    from transformers import TrainerCallback
except ImportError:  # pragma: no cover - lets the module be imported/tested without transformers
    TrainerCallback = object

from edgevlm.constants import LABELS


def balanced_validation_subset(val_df, samples_per_class: int, labels: Sequence[str] = LABELS, seed: int = 0):
    """Fixed class-balanced subset (sampled once, so every epoch is scored on identical rows).

    A class with fewer than ``samples_per_class`` rows contributes all its rows.
    """
    import pandas as pd

    parts = []
    for label in labels:
        class_rows = val_df[val_df["label"] == label]
        n = min(samples_per_class, len(class_rows))
        parts.append(class_rows.sample(n=n, random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


class MacroF1EarlyStopping(TrainerCallback):
    """Select checkpoints by validation macro-F1 computed from real generations.

    ``predict_fn(model, processor, check_df) -> list[str]`` returns one predicted label per
    row of ``check_df`` (use ``edgevlm.evaluation.inference.predict_batch_rows`` via
    ``functools.partial``).
    """

    def __init__(
        self,
        processor,
        check_df,
        output_dir,
        predict_fn: Callable,
        patience: int = 3,
        resume: bool = False,
        labels: Sequence[str] = LABELS,
    ) -> None:
        self.processor = processor
        self.check_df = check_df
        self.output_dir = Path(output_dir)
        self.predict_fn = predict_fn
        self.patience = patience
        self.labels = list(labels)
        self.best_macro_f1 = -1.0
        self.epochs_without_improvement = 0
        self.history: list[dict] = []
        self.log_path = self.output_dir / "macro_f1_epoch_log.json"

        # When training continues from a checkpoint the bookkeeping must continue too; otherwise
        # "best so far" restarts at -1 and the first epoch after the restart overwrites
        # best_by_macro_f1 with a worse model than the one already saved.
        if resume and self.log_path.is_file():
            try:
                self.history = json.loads(self.log_path.read_text())
            except (OSError, ValueError):
                self.history = []
            if self.history:
                scores = [record["macro_f1"] for record in self.history]
                best_index = max(range(len(scores)), key=lambda index: scores[index])
                self.best_macro_f1 = scores[best_index]
                self.epochs_without_improvement = len(scores) - 1 - best_index
                print(
                    f"[MacroF1EarlyStopping] restored {len(scores)} earlier epoch result(s); "
                    f"best macro-F1 so far {self.best_macro_f1:.4f}, "
                    f"{self.epochs_without_improvement} epoch(s) without improvement."
                )

    def on_epoch_end(self, args, state, control, **kwargs):
        from sklearn.metrics import f1_score

        model_ref = kwargs.get("model", None)
        if model_ref is None:
            print("[MacroF1EarlyStopping] no model in callback kwargs - skipping this epoch's check.")
            return control

        try:
            model_ref.eval()
            t0 = time.perf_counter()
            predicted = self.predict_fn(model_ref, self.processor, self.check_df)
            true = [str(label).strip().upper() for label in self.check_df["label"].tolist()]
            elapsed = time.perf_counter() - t0

            macro_f1 = f1_score(true, predicted, labels=self.labels, average="macro", zero_division=0)
            per_class = f1_score(true, predicted, labels=self.labels, average=None, zero_division=0)

            improved = macro_f1 > self.best_macro_f1
            if improved:
                self.best_macro_f1 = macro_f1
                self.epochs_without_improvement = 0
                best_dir = self.output_dir / "best_by_macro_f1"
                best_dir.mkdir(parents=True, exist_ok=True)
                model_ref.save_pretrained(str(best_dir))
                self.processor.save_pretrained(str(best_dir))
            else:
                self.epochs_without_improvement += 1

            self.history.append(
                {
                    "epoch": state.epoch,
                    "macro_f1": macro_f1,
                    "per_class_f1": dict(zip(self.labels, per_class.tolist())),
                    "improved": improved,
                    "eval_seconds": elapsed,
                }
            )
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "w") as handle:
                json.dump(self.history, handle, indent=2)

            print(
                f"\n[MacroF1EarlyStopping] epoch {state.epoch:.2f}: macro_f1={macro_f1:.4f} "
                f"(best={self.best_macro_f1:.4f}) per_class_f1={dict(zip(self.labels, np.round(per_class, 3)))} "
                f"[{elapsed:.1f}s on {len(self.check_df)} samples]"
            )

            if self.epochs_without_improvement >= self.patience:
                print(
                    f"[MacroF1EarlyStopping] no improvement for {self.patience} epochs - stopping training. "
                    f"Best checkpoint saved at: {self.output_dir / 'best_by_macro_f1'}"
                )
                control.should_training_stop = True

        except Exception as error:  # a failed check must not kill a multi-hour run
            print(
                "[MacroF1EarlyStopping] check failed this epoch (training continues): "
                f"{type(error).__name__}: {error}"
            )
        finally:
            model_ref.train()

        return control


class RunMonitor(TrainerCallback):
    """Append one plain-text line per logging step (step, epoch, loss, lr, process/GPU memory).

    Lets a multi-day run be followed with ``tail -f`` and shows what the run was doing right
    before a server drop, which the notebook output area cannot (it vanishes with the kernel).
    Monitoring must never be the reason a long run stops, so nothing here may raise.
    """

    def __init__(self, log_path) -> None:
        self.log_path = Path(log_path)
        try:
            import psutil

            self._process = psutil.Process()
        except Exception:
            self._process = None

    def _write(self, text: str) -> None:
        try:
            with open(self.log_path, "a") as handle:
                handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}\n")
        except OSError:
            pass

    def _memory(self) -> tuple[float, float]:
        rss_mb = self._process.memory_info().rss / 1e6 if self._process is not None else float("nan")
        try:
            import torch

            gpu_mb = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
        except Exception:
            gpu_mb = 0.0
        return rss_mb, gpu_mb

    def on_train_begin(self, args, state, control, **kwargs):
        self._write(f"train_begin global_step={state.global_step} max_steps={state.max_steps}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        try:
            logs = logs or {}
            rss_mb, gpu_mb = self._memory()
            epoch = state.epoch if state.epoch is not None else 0.0
            parts = [f"step={state.global_step}/{state.max_steps}", f"epoch={epoch:.3f}"]
            for key in ("loss", "eval_loss", "learning_rate", "grad_norm"):
                if key in logs:
                    parts.append(f"{key}={logs[key]:.6g}")
            parts += [f"rss_mb={rss_mb:.0f}", f"gpu_mb={gpu_mb:.0f}"]
            self._write(" ".join(parts))
        except Exception:
            pass

    def on_save(self, args, state, control, **kwargs):
        self._write(f"checkpoint_saved step={state.global_step}")

    def on_train_end(self, args, state, control, **kwargs):
        self._write(f"train_end global_step={state.global_step}")
