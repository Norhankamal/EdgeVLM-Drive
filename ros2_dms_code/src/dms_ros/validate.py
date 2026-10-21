#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_FLAX"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import cv2
import numpy as np
import yaml
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parent
FRAME_ROOT = Path("/home/jetson/test_dataset")
CONFIG_PATH = PROJECT_ROOT / "config" / "dms_params.yaml"
PROMPTS_PATH = PROJECT_ROOT / "config" / "prompts.yaml"
RESULTS_DIR = PROJECT_ROOT / "validation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["NORMAL", "DROWSY", "LOOKING_AWAY", "PHONE_USAGE"]
UNSAFE_LABELS = ["DROWSY", "LOOKING_AWAY", "PHONE_USAGE"]
LIMIT_PER_CLASS = int(os.environ.get("DMS_LIMIT", 0)) or None
SUFFIX = f"_limit{LIMIT_PER_CLASS}" if LIMIT_PER_CLASS else ""
SAVE_EVERY = 50


def load_config(path):
    with open(path, "r") as handle:
        data = yaml.safe_load(handle)
    with open(PROMPTS_PATH, "r") as handle:
        prompts = yaml.safe_load(handle)
    return data["inference_node"]["ros__parameters"], prompts


PARAMS, VALIDATE_SECTION = load_config(CONFIG_PATH)
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else PARAMS["active_model"]
MODEL_DIR = Path(PARAMS["models_dir"]) / MODEL_NAME
IMAGE_SIZE = int(PARAMS["longest_edge"])
MAX_NEW_TOKENS = int(PARAMS["max_new_tokens"])
DO_SAMPLE = bool(PARAMS["do_sample"])

MODEL_PROMPTS = VALIDATE_SECTION.get("model_prompts", {})
if MODEL_NAME not in MODEL_PROMPTS:
    raise KeyError(
        f"No prompt for '{MODEL_NAME}' in {PROMPTS_PATH}. "
        f"Available: {list(MODEL_PROMPTS)}"
    )
INSTRUCTION = str(MODEL_PROMPTS[MODEL_NAME]).strip()

PREDICTIONS_CSV = RESULTS_DIR / f"{MODEL_NAME}_dualcrop_predictions{SUFFIX}.csv"
SUMMARY_CSV = RESULTS_DIR / f"{MODEL_NAME}_dualcrop_summary{SUFFIX}.csv"


import dms_preprocess as lgp

if lgp._get_haar_face_detector().empty() or lgp._get_haar_profile_detector().empty():
    raise RuntimeError("Haar face cascades failed to load. Face crops would be wrong.")
print("OpenCV:", cv2.__version__, "| cascades:", lgp._cascade_dir())


def create_dual_crops(image_path):
    raw_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if raw_bgr is None:
        raise FileNotFoundError(f"OpenCV could not read: {image_path}")
    enhanced_rgb, _ = lgp.enhance_frame(raw_bgr)
    full_crop = lgp.resize_full_frame(enhanced_rgb)
    face_crop = lgp.extract_face_crop(enhanced_rgb, landmarks=None)
    return full_crop, face_crop


def collect_dataset(root, limit_per_class=None):
    rows = []
    for label in LABELS:
        label_dir = Path(root) / label
        if not label_dir.is_dir():
            print(f"WARNING: folder not found: {label_dir}")
            continue
        paths = [
            p for p in sorted(label_dir.rglob("*"))
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if limit_per_class and len(paths) > limit_per_class:
            picks = np.linspace(0, len(paths) - 1, limit_per_class).round().astype(int)
            paths = [paths[i] for i in picks]
        rows.extend({"frame_path": str(p), "label": label} for p in paths)
    return pd.DataFrame(rows)


def parse_prediction(raw_text):
    normalized = str(raw_text).strip().upper()
    exact = normalized in LABELS
    cleaned = normalized.strip(" .,;:!?'\"")
    return (cleaned if cleaned in LABELS else "INVALID"), exact


def load_model_stack(model_dir, device, dtype):
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model = model.to(device)
    model.eval()
    model.config.use_cache = True
    return processor, model


def infer_single(processor, model, image_path, device):
    full_crop, face_crop = create_dual_crops(image_path)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "image"},
            {"type": "text", "text": INSTRUCTION},
        ],
    }]
    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=prompt,
        images=[[Image.fromarray(full_crop).convert("RGB"),
                 Image.fromarray(face_crop).convert("RGB")]],
        return_tensors="pt",
        size={"longest_edge": IMAGE_SIZE},
        max_image_size={"longest_edge": IMAGE_SIZE},
    )
    inputs = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }

    pixel_values = inputs["pixel_values"]
    if pixel_values.ndim != 5 or pixel_values.shape[1] != 2:
        raise RuntimeError(f"Expected 2 images per sample, got {tuple(pixel_values.shape)}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=DO_SAMPLE
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - started) * 1000

    input_length = inputs["input_ids"].shape[-1]
    raw = processor.batch_decode(
        generated[:, input_length:], skip_special_tokens=True
    )[0].strip()
    prediction, exact = parse_prediction(raw)
    return raw, prediction, exact, latency_ms


def load_saved_rows(wanted):
    if not PREDICTIONS_CSV.is_file():
        return []
    try:
        saved = pd.read_csv(PREDICTIONS_CSV)
    except pd.errors.EmptyDataError:
        return []
    if "frame_path" not in saved.columns:
        return []
    return [row for row in saved.to_dict("records") if row["frame_path"] in wanted]


def run_evaluation(processor, model, eval_df, device):
    records = load_saved_rows(set(eval_df["frame_path"]))
    finished = {row["frame_path"] for row in records}
    print("Resuming after", len(finished), "images.")

    remaining = eval_df[~eval_df["frame_path"].isin(finished)]
    for completed, (_, row) in enumerate(
        tqdm(remaining.iterrows(), total=len(remaining), desc="Evaluating"), start=1
    ):
        try:
            raw, prediction, exact, latency_ms = infer_single(
                processor, model, row["frame_path"], device
            )
        except Exception:
            print("Failed on:", row["frame_path"], flush=True)
            if records:
                pd.DataFrame(records).to_csv(PREDICTIONS_CSV, index=False)
            raise
        records.append({
            "frame_path": row["frame_path"],
            "true_label": row["label"],
            "raw_output": raw,
            "prediction": prediction,
            "exact_label_valid": exact,
            "latency_ms": latency_ms,
        })
        if completed % SAVE_EVERY == 0:
            pd.DataFrame(records).to_csv(PREDICTIONS_CSV, index=False)

    predictions_df = pd.DataFrame(records).drop_duplicates("frame_path", keep="last")
    predictions_df.to_csv(PREDICTIONS_CSV, index=False)
    return predictions_df


def summarize(predictions_df):
    y_true = predictions_df["true_label"].astype(str).to_numpy()
    y_pred = predictions_df["prediction"].astype(str).to_numpy()
    latencies = predictions_df["latency_ms"].astype(float).to_numpy()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    unsafe_true = np.isin(y_true, UNSAFE_LABELS)
    normal_true = y_true == "NORMAL"
    unsafe_pred = np.isin(y_pred, UNSAFE_LABELS)

    unsafe_to_safe_miss = (
        np.mean(np.isin(y_pred[unsafe_true], ["NORMAL", "INVALID"])) if unsafe_true.any() else 0.0
    )
    unsafe_class_error = (
        np.mean(y_pred[unsafe_true] != y_true[unsafe_true]) if unsafe_true.any() else 0.0
    )
    false_alarm = np.mean(unsafe_pred[normal_true]) if normal_true.any() else 0.0
    exact_validity = predictions_df["exact_label_valid"].astype(bool).mean()
    parser_failure = np.mean(y_pred == "INVALID")
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)

    matrix_labels = LABELS + ["INVALID"]
    cm = confusion_matrix(y_true, y_pred, labels=matrix_labels)

    print("\n==========", MODEL_NAME, "- DUAL-CROP JETSON TEST ==========")
    print("Images:", len(predictions_df))
    print("Accuracy:", round(accuracy_score(y_true, y_pred) * 100, 2), "%")
    print("Balanced accuracy:", round(recall.mean() * 100, 2), "%")
    print("Macro-F1:", round(macro_f1 * 100, 2), "%")
    print("Exact-label validity:", round(exact_validity * 100, 2), "%")
    print("Parser failure:", round(parser_failure * 100, 2), "%")
    print("\nPer-class results:")
    for label, p, r, score, count in zip(LABELS, precision, recall, f1, support):
        print(f"{label}: precision={p:.3f}, recall={r:.3f}, F1={score:.3f}, samples={count}")
    print("\nConfusion matrix:")
    print(pd.DataFrame(cm, index=matrix_labels, columns=matrix_labels))
    print("Unsafe-to-safe miss rate:", round(unsafe_to_safe_miss * 100, 2), "%")
    print("Unsafe-class error rate:", round(unsafe_class_error * 100, 2), "%")
    print("False-alarm rate:", round(false_alarm * 100, 2), "%")
    print("Mean latency (ms):", round(float(np.mean(latencies)), 2))
    print("P95 latency (ms):", round(float(np.percentile(latencies, 95)), 2))
    if torch.cuda.is_available():
        print("Peak GPU memory (MB):", round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2))

    summary = {
        "images": len(predictions_df),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(macro_f1),
        "exact_label_validity": float(exact_validity),
        "parser_failure": float(parser_failure),
        "unsafe_to_safe_miss": float(unsafe_to_safe_miss),
        "unsafe_class_error": float(unsafe_class_error),
        "false_alarm": float(false_alarm),
        "mean_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
    }
    pd.DataFrame([summary]).to_csv(SUMMARY_CSV, index=False)
    print("\nSummary saved to:", SUMMARY_CSV)


def main():
    eval_df = collect_dataset(FRAME_ROOT, LIMIT_PER_CLASS)
    if len(eval_df) == 0:
        raise RuntimeError(f"No images found under {FRAME_ROOT}. Expected folders: {LABELS}")

    print("Scanning:", FRAME_ROOT)
    for label in LABELS:
        print(f"{label:12s}: {int((eval_df['label'] == label).sum()):5d} images")
    print("Total:", len(eval_df), "images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print("Device:", device, "| dtype:", dtype)

    processor, model = load_model_stack(MODEL_DIR, device, dtype)
    print("Model loaded:", MODEL_DIR)
    print("Config:", CONFIG_PATH, "| model:", MODEL_NAME, "| size:", IMAGE_SIZE, "| max_new_tokens:", MAX_NEW_TOKENS, "| do_sample:", DO_SAMPLE)

    print("Warming up...")
    infer_single(processor, model, eval_df.iloc[0]["frame_path"], device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    predictions_df = run_evaluation(processor, model, eval_df, device)
    summarize(predictions_df)


if __name__ == "__main__":
    main()
