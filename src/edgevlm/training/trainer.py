"""Class-balanced training: weighted sampler, Trainer subclass and TrainingArguments builder.

Sampling weight of an example of class *c* is ``1 / count(c) ** power``:

* ``power = 1.0``  full inverse-frequency weighting (Phase 1 baseline and Phase 2 v1);
* ``power = 0.5``  square-root damping (Phase 2 v2 / final). Full inverse frequency
  over-corrects at this dataset's class ratios - the rarest class is seen far more often than
  its real prevalence, which pushes the model to over-predict it (extra false alarms).
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

try:
    from transformers import Trainer as _Trainer
except ImportError:  # pragma: no cover
    _Trainer = object


def inverse_frequency_weights(labels: Sequence[str], power: float = 0.5) -> list[float]:
    """Per-example sampling weights ``1 / count(label) ** power``."""
    counts = Counter(labels)
    return [1.0 / (counts[label] ** power) for label in labels]


class BalancedTrainer(_Trainer):
    """``Trainer`` that draws each epoch with replacement using class-balancing weights.

    Set ``BalancedTrainer.sampler_weight_power`` (class attribute) before instantiation;
    the train dataset must expose ``.dataframe`` with a ``label`` column.
    """

    sampler_weight_power: float = 0.5

    def _get_train_sampler(self, *args, **kwargs):  # signature differs across transformers versions
        import torch
        from torch.utils.data import WeightedRandomSampler

        labels = self.train_dataset.dataframe["label"].astype(str).tolist()
        weights = torch.tensor(inverse_frequency_weights(labels, self.sampler_weight_power), dtype=torch.double)
        return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def build_training_args(
    output_dir,
    epochs: int,
    learning_rate: float = 1e-4,
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.0,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    logging_steps: int = 50,
    save_strategy: str = "steps",
    save_steps: int = 100,
    save_total_limit: int = 3,
    load_best_model_at_end_by_eval_loss: bool = False,
):
    """``TrainingArguments`` shared by all runs (transformers 4.46 argument names).

    ``save_strategy="epoch"`` with ``load_best_model_at_end_by_eval_loss=True`` reproduces the
    Phase 1 baseline; note that the deployable checkpoint is always
    ``<output_dir>/best_by_macro_f1`` written by ``MacroF1EarlyStopping``, not the model that
    ``Trainer`` reloads at the end.
    """
    import torch
    from transformers import TrainingArguments

    kwargs = dict(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=epochs,  # upper bound only - macro-F1 early stopping decides the real stop
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=weight_decay,
        fp16=torch.cuda.is_available(),
        logging_steps=logging_steps,
        eval_strategy="epoch",
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        optim="adamw_torch",
        report_to="none",
    )
    if save_strategy == "steps":
        kwargs["save_steps"] = save_steps
    if load_best_model_at_end_by_eval_loss:
        kwargs.update(load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False)
    return TrainingArguments(**kwargs)
