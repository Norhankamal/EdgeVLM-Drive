#!/usr/bin/env python
"""Evaluate a merged FP16 package on a class-balanced subset: accuracy, per-class results, latency, peak GPU memory.

    python scripts/benchmark_jetson.py --package export/smolvlm_500m --csv test.csv --data-root /path/to/data \
        --per-class 300 --instruction v2 --out results/jetson/run.json

The model runs in FP16 on the GPU; images go through the same dual-crop preprocessing and strict
label parser as the training-time evaluation. Run it on the device whose latency you want to report
(for the Jetson Nano results, in the 5 W power mode).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _common  # noqa: F401  (sets environment variables before torch is imported)
import pandas as pd
import torch

from edgevlm import constants
from edgevlm.data.frame_paths import FrameResolver
from edgevlm.edge.benchmark import balanced_subset, peak_gpu_memory_mb, summarise_latency
from edgevlm.edge.stats import wilson_interval
from edgevlm.evaluation.inference import predict_row
from edgevlm.evaluation.metrics import compute_metrics
from edgevlm.pipeline import get_preprocessing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True, help="merged FP16 package directory")
    ap.add_argument("--csv", required=True, help="CSV with frame_path and label columns")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--instruction", choices=["v1", "v2"], default="v2",
                    help="must match the prompt the model was trained with")
    ap.add_argument("--preprocessing", default="v2", choices=["v1", "v2"])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--predictions-out", default=None, help="optional CSV of per-image predictions")
    args = ap.parse_args()

    from transformers import AutoModelForVision2Seq, AutoProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.package)
    model = AutoModelForVision2Seq.from_pretrained(args.package, torch_dtype=dtype).to(device).eval()

    df = balanced_subset(pd.read_csv(args.csv), args.per_class)
    resolver = FrameResolver(Path(args.data_root), Path(args.data_root), Path(args.data_root))
    preprocessing = get_preprocessing(args.preprocessing)
    instruction = constants.INSTRUCTION_V2 if args.instruction == "v2" else constants.INSTRUCTION_V1

    def run(row):
        return predict_row(model, processor, row, "test", resolver, preprocessing, instruction)

    for _, row in df.head(args.warmup).iterrows():  # untimed warm-up (CUDA kernels, allocator)
        run(row)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    records = []
    for _, row in df.iterrows():
        raw, pred, valid, latency_ms, path = run(row)
        records.append({"frame_path": path, "true": row["label"], "raw_output": raw, "pred": pred,
                        "valid": valid, "latency_ms": latency_ms})
    res = pd.DataFrame(records)

    metrics = compute_metrics(res["true"], res["pred"], res["valid"], res["latency_ms"])
    correct = int((res["true"] == res["pred"]).sum())
    lo, hi = wilson_interval(correct, len(res))
    metrics["accuracy_ci95"] = [lo, hi]
    metrics["latency"] = summarise_latency(res["latency_ms"])
    metrics["peak_gpu_memory_mb"] = peak_gpu_memory_mb()
    metrics["package"] = str(args.package)
    Path(args.out).write_text(json.dumps(metrics, indent=2))
    if args.predictions_out:
        res.to_csv(args.predictions_out, index=False)
    print(json.dumps({k: metrics[k] for k in ("n", "accuracy", "macro_f1", "latency", "peak_gpu_memory_mb")}, indent=2))


if __name__ == "__main__":
    main()
