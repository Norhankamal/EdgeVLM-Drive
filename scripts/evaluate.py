#!/usr/bin/env python
"""Evaluate a LoRA adapter on a split, one image at a time (same protocol as the notebooks).

    python scripts/evaluate.py --config configs/smolvlm500m_phase2_final.yaml \
        --adapter runs/smolvlm500m_phase2_final/best_by_macro_f1 --split test

Writes ``metrics.json`` and ``predictions.csv`` next to the adapter's run directory (or --out).
Latency is measured on whatever device this runs on (not the Jetson).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _common as common
import torch

from edgevlm.config import load_config
from edgevlm.evaluation.inference import evaluate_split
from edgevlm.evaluation.metrics import format_report
from edgevlm.pipeline import get_preprocessing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", required=True, help="adapter directory (e.g. .../best_by_macro_f1)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", default=None, help="output directory (default: <adapter>/../eval_<split>)")
    args = ap.parse_args()

    from peft import PeftModel
    from tqdm.auto import tqdm
    from transformers import AutoModelForVision2Seq, AutoProcessor

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.adapter)
    base = AutoModelForVision2Seq.from_pretrained(cfg.model.model_id, cache_dir=str(cfg.model_cache), torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, args.adapter).to(device).eval()

    df = common.read_split(cfg, args.split)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    split_name = "validation" if args.split == "val" else args.split
    metrics, results = evaluate_split(model, processor, df, split_name, common.make_resolver(cfg),
                                      get_preprocessing(cfg.data.preprocessing), cfg.instruction_text,
                                      cfg.model.image_size, progress=tqdm)
    if torch.cuda.is_available():
        metrics["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1024**2
    print(format_report(metrics, f"{args.split.upper()} RESULTS ({cfg.name})"))

    out = Path(args.out) if args.out else Path(args.adapter).parent / f"eval_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    results.to_csv(out / "predictions.csv", index=False)
    print("Saved to", out)


if __name__ == "__main__":
    main()
