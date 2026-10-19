#!/usr/bin/env python
"""Fine-tune SmolVLM-500M (dual-crop, vision+text LoRA, macro-F1 checkpoint selection).

    python scripts/train.py --config configs/smolvlm500m_phase2_final.yaml [--smoke-test] [--no-resume]

Writes to ``<output_root>/<name>/``: ``best_by_macro_f1/`` (the adapter to deploy),
``macro_f1_epoch_log.json`` and ``training_progress.log``.
"""
from __future__ import annotations

import argparse
import json
from functools import partial

import _common as common
import torch

from edgevlm.config import load_config
from edgevlm.data.dataset import DriverStateDualCropDataset, SmolVLMDualCropCollator
from edgevlm.evaluation.inference import predict_batch_rows
from edgevlm.modeling.lora import attach_lora
from edgevlm.pipeline import get_preprocessing
from edgevlm.training.callbacks import MacroF1EarlyStopping, RunMonitor, balanced_validation_subset
from edgevlm.training.resume import allow_trusted_torch_load, find_resume_checkpoint
from edgevlm.training.trainer import BalancedTrainer, build_training_args


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke-test", action="store_true", help="run 3 optimizer steps and exit")
    ap.add_argument("--no-resume", action="store_true", help="ignore existing checkpoints in the output directory")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    preprocessing = get_preprocessing(cfg.data.preprocessing)
    resolver = common.make_resolver(cfg)

    train_df, val_df = common.read_split(cfg, "train"), common.read_split(cfg, "val")
    common.check_images_exist(train_df, "train", resolver)
    common.check_images_exist(val_df, "validation", resolver)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor, model = common.load_base_model(cfg, device, dtype)
    model = attach_lora(model, cfg.lora.r, cfg.lora.alpha, cfg.lora.dropout)

    make = lambda df, split: DriverStateDualCropDataset(df, split, resolver, preprocessing, cfg.instruction_text)
    train_ds, val_ds = make(train_df, "train"), make(val_df, "validation")
    collator = SmolVLMDualCropCollator(processor, cfg.model.image_size)
    t = cfg.training

    if args.smoke_test:
        from transformers import Trainer, TrainingArguments

        smoke = TrainingArguments(output_dir=str(out / "smoke_test"), per_device_train_batch_size=1,
                                  gradient_accumulation_steps=4, max_steps=3, learning_rate=t.learning_rate,
                                  fp16=torch.cuda.is_available(), logging_steps=1, save_strategy="no",
                                  eval_strategy="no", report_to="none", remove_unused_columns=False,
                                  dataloader_num_workers=0, optim="adamw_torch")
        print(Trainer(model=model, args=smoke, train_dataset=train_ds, data_collator=collator).train())
        return

    # ---- resume logic -------------------------------------------------------------------
    allow_trusted_torch_load()
    resume_ckpt = None if args.no_resume else find_resume_checkpoint(out)
    best_adapter = out / "best_by_macro_f1" / "adapter_model.safetensors"
    epoch_log = out / "macro_f1_epoch_log.json"
    epochs_this_run, continuing = t.epochs, False
    if resume_ckpt is not None:
        continuing = True
        print(f"Continuing the earlier run from {resume_ckpt.name} (full training state).")
    elif t.warm_start_from_best and not args.no_resume and best_adapter.is_file():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(model, load_file(str(best_adapter)))
        continuing = True
        done = len(json.loads(epoch_log.read_text())) if epoch_log.is_file() else 0
        epochs_this_run = max(1, t.epochs - done)
        print(f"Warm start from saved best adapter ({done} epoch(s) behind it); training {epochs_this_run} more.")
    else:
        print("No earlier run found - training from step 0.")

    vc = cfg.validation_check
    check_df = balanced_validation_subset(val_df, vc.samples_per_class, seed=vc.seed)
    print("Per-epoch macro-F1 check subset:", len(check_df), check_df["label"].value_counts().to_dict())
    predict_fn = partial(predict_batch_rows, split_name="validation", resolver=resolver, preprocessing=preprocessing,
                         instruction=cfg.instruction_text, image_size=cfg.model.image_size, batch_size=vc.batch_size)
    early_stop = MacroF1EarlyStopping(processor, check_df, out, lambda m, p, df: predict_fn(m, p, df),
                                      patience=vc.patience, resume=continuing)

    BalancedTrainer.sampler_weight_power = t.sampler_weight_power
    trainer = BalancedTrainer(
        model=model,
        args=build_training_args(out, epochs_this_run, t.learning_rate, t.warmup_ratio, t.weight_decay,
                                 t.per_device_batch_size, t.gradient_accumulation_steps, t.logging_steps,
                                 t.save_strategy, t.save_steps, t.save_total_limit, t.trainer_loads_best_by_eval_loss),
        train_dataset=train_ds, eval_dataset=val_ds, data_collator=collator,
        callbacks=[early_stop, RunMonitor(out / "training_progress.log")],
    )
    print(trainer.train(resume_from_checkpoint=str(resume_ckpt) if resume_ckpt else None))
    trainer.save_model(str(out))
    processor.save_pretrained(str(out))
    print(f"Last-epoch adapter: {out}\nDeployable (macro-F1-best) adapter: {out / 'best_by_macro_f1'}")


if __name__ == "__main__":
    main()
