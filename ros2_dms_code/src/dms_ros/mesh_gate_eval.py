#!/usr/bin/env python3
"""
Per-image evaluation of the mesh gate on a labelled frame folder (Jetson).

Run it after the model predictions exist in RESULTS_DIR
(`<model>_dualcrop_predictions.csv`, columns frame_path, prediction, latency_ms).

What it does
  1. Extracts, for every image, the mesh readings and the full mesh feature
     vector with ONE reused extractor (warm timing) and saves them to
     validation_results/mesh_signals.csv.
  2. Applies the real gate nodes of the pipeline (`mesh_signals_node` and
     `decision_gate_node` from the pipeline module) twice:
       - rules only      (EAR / yaw)
       - rules + any-doubt triage (the pipeline default)
     with the scheduled check switched off, because single images carry no
     elapsed time. PHONE_USAGE numbers are therefore the worst case: in a
     live stream the scheduled check bounds how long a phone can go unseen.
  3. Prints and saves, per model, accuracy / balanced accuracy / macro-F1 of
     "model alone" versus "mesh + model", recall per class, share of model
     calls skipped, unsafe frames wrongly skipped, and the estimated time per
     frame.

Send back the whole `validation_results` folder (mesh_signals.csv,
mesh_gate_summary.csv, the prediction CSVs).
"""
import importlib
import time
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tqdm.auto import tqdm

# Name under which the pipeline module is installed on the Jetson.
PIPELINE_MODULE = "dms_preprocess"
lgp = importlib.import_module(PIPELINE_MODULE)

PROJECT_ROOT = Path(__file__).resolve().parent
FRAME_ROOT = Path("/home/jetson/test_dataset")
RESULTS_DIR = PROJECT_ROOT / "validation_results"
LIMIT_PER_CLASS = int(os.environ.get("DMS_LIMIT", 0)) or None
SUFFIX = f"_limit{LIMIT_PER_CLASS}" if LIMIT_PER_CLASS else ""
SIGNALS_CSV = RESULTS_DIR / f"mesh_signals{SUFFIX}.csv"
SUMMARY_CSV = RESULTS_DIR / f"mesh_gate_summary{SUFFIX}.csv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["NORMAL", "DROWSY", "LOOKING_AWAY", "PHONE_USAGE"]
UNSAFE_LABELS = LABELS[1:]
MODELS = ["SmolVLM-256M-Finetuned", "SmolVLM-500M-Finetuned"]

SKIP_REASON = "confidently_normal"
SIM_EPOCH = 1.7e9
GATES = {"rules only": None, "rules + any-doubt triage": "default"}


def collect_dataset(root, limit_per_class=None):
    rows = []
    for label in LABELS:
        label_dir = Path(root) / label
        if not label_dir.is_dir():
            print(f"WARNING: folder not found: {label_dir}")
            continue
        paths = [p for p in sorted(label_dir.rglob("*"))
                 if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if limit_per_class and len(paths) > limit_per_class:
            picks = np.linspace(0, len(paths) - 1, limit_per_class).round().astype(int)
            paths = [paths[i] for i in picks]
        rows.extend({"frame_path": str(p), "true_label": label} for p in paths)
    return pd.DataFrame(rows)


def compute_signals(eval_df, warmup=5, repeats=3):
    from mesh_landmark_extractor import FEATURE_NAMES, MeshLandmarkExtractor

    rows = []
    with MeshLandmarkExtractor() as extractor:
        first = cv2.imread(eval_df.iloc[0]["frame_path"], cv2.IMREAD_COLOR)
        warm_rgb, _ = lgp.enhance_frame(first)
        for _ in range(warmup):
            extractor.extract(warm_rgb)
        for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Mesh"):
            raw_bgr = cv2.imread(row["frame_path"], cv2.IMREAD_COLOR)
            frame_rgb, quality_ok = lgp.enhance_frame(raw_bgr)
            times = []
            for _ in range(repeats):
                started = time.perf_counter()
                result = extractor.extract(frame_rgb)
                times.append((time.perf_counter() - started) * 1000)
            found = result.to_landmarks_dict() is not None
            features = (result.features or {}) if found else {}
            record = {
                "frame_path": row["frame_path"], "true_label": row["true_label"], "face_found": found,
                "quality_ok": True if quality_ok is None else bool(quality_ok),
                "mesh_ms": float(np.median(times)),
            }
            for name in FEATURE_NAMES:
                record[name] = features.get(name, np.nan)
            rows.append(record)
    signals = pd.DataFrame(rows)
    signals.to_csv(SIGNALS_CSV, index=False)
    return signals


def load_signals(eval_df):
    if SIGNALS_CSV.is_file():
        try:
            saved = pd.read_csv(SIGNALS_CSV)
        except pd.errors.EmptyDataError:
            saved = None
        from mesh_landmark_extractor import FEATURE_NAMES
        if (saved is not None and set(saved["frame_path"]) == set(eval_df["frame_path"])
                and all(name in saved.columns for name in FEATURE_NAMES)):
            print("Reusing saved mesh readings:", SIGNALS_CSV)
            return saved
    return compute_signals(eval_df)


def landmarks_of(row, feature_names):
    if not row["face_found"] or pd.isna(row["ear"]) or pd.isna(row["yaw"]):
        return None
    return {"eye_aspect_ratio": float(row["ear"]), "head_yaw_deg": float(row["yaw"]),
            "tracking_confidence": 1.0,
            "mesh_features": {name: float(row[name]) for name in feature_names}}


def apply_gate(signals, triage):
    """Run the pipeline's own gate nodes on every image (scheduled check off)."""
    from mesh_landmark_extractor import FEATURE_NAMES

    original = getattr(lgp, "MESH_TRIAGE", None)
    default = original if triage == "default" else None
    lgp.set_mesh_triage(default)
    reasons = []
    try:
        for _, row in signals.iterrows():
            state = lgp.new_frame_state(
                0, None, landmarks_of(row, FEATURE_NAMES), last_vlm_timestamp=SIM_EPOCH,
                scheduled_interval_s=1e12, timestamp=SIM_EPOCH + 1.0)
            state["frame_quality_ok"] = bool(row["quality_ok"])
            state.update(lgp.mesh_signals_node(state))
            reasons.append(lgp.decision_gate_node(state)["gate_reason"])
    finally:
        lgp.set_mesh_triage(original)
    return np.array(reasons)


def print_gate_table(signals, reason, title):
    print(f"\nGate decisions per class - {title}")
    true_label = signals["true_label"].to_numpy()
    reason_names = sorted(set(reason))
    print(f"{'class':14s}{'frames':>8s}{'skip%':>8s}  " + "  ".join(f"{r[:14]:>14s}" for r in reason_names if r != SKIP_REASON))
    for label in LABELS:
        in_class = true_label == label
        total = max(int(in_class.sum()), 1)
        cells = "  ".join(f"{np.mean(reason[in_class] == r) * 100:13.1f}%" for r in reason_names if r != SKIP_REASON)
        print(f"{label:14s}{int(in_class.sum()):8d}{np.mean(reason[in_class] == SKIP_REASON) * 100:8.1f}  {cells}")


def score(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return {"accuracy": float(np.mean(y_true == y_pred)), "balanced_acc": float(recall.mean()),
            "macro_f1": float(f1.mean()), "recall": recall}


def evaluate_model(name, gate_name, signals, reason):
    path = RESULTS_DIR / f"{name}_dualcrop_predictions{SUFFIX}.csv"
    if not path.is_file():
        print(f"\n{name}: no predictions file, skipped.")
        return None
    predictions = pd.read_csv(path)[["frame_path", "prediction", "latency_ms"]]
    merged = signals.assign(reason=reason).merge(predictions, on="frame_path", how="inner")
    if len(merged) < len(signals):
        print(f"\n{name}: only {len(merged)} of {len(signals)} predictions saved, skipped.")
        return None

    y_true = merged["true_label"].to_numpy()
    model_pred = merged["prediction"].to_numpy()
    skipped = (merged["reason"] == SKIP_REASON).to_numpy()
    system_pred = np.where(skipped, "NORMAL", model_pred)

    alone, combined = score(y_true, model_pred), score(y_true, system_pred)
    unsafe = np.isin(y_true, UNSAFE_LABELS)
    unsafe_skipped = int(np.sum(unsafe & skipped))
    skip_rate = float(np.mean(skipped))
    model_ms, mesh_ms = float(merged["latency_ms"].mean()), float(merged["mesh_ms"].mean())
    estimated_ms = (1.0 - skip_rate) * model_ms + mesh_ms

    print(f"\n========== {name} | gate: {gate_name} ==========")
    print(f"{'':16s}{'accuracy':>10s}{'balanced':>10s}{'macro-F1':>10s}")
    print(f"{'Model alone':16s}{alone['accuracy'] * 100:9.2f}%{alone['balanced_acc'] * 100:9.2f}%{alone['macro_f1'] * 100:9.2f}%")
    print(f"{'Mesh + model':16s}{combined['accuracy'] * 100:9.2f}%{combined['balanced_acc'] * 100:9.2f}%{combined['macro_f1'] * 100:9.2f}%")
    print("\nRecall per class (model alone -> mesh + model):")
    for label, before, after in zip(LABELS, alone["recall"], combined["recall"]):
        print(f"  {label:14s}{before:.3f} -> {after:.3f}")
    print(f"\nModel calls skipped: {int(skipped.sum())} of {len(merged)} ({skip_rate * 100:.1f}%)")
    print(f"Unsafe frames wrongly skipped: {unsafe_skipped} of {int(unsafe.sum())} "
          f"({unsafe_skipped / max(int(unsafe.sum()), 1) * 100:.1f}%)")
    for label in UNSAFE_LABELS:
        m = y_true == label
        print(f"  {label:14s}skipped {int((skipped & m).sum())} of {int(m.sum())}")
    matrix_labels = LABELS + ["INVALID"]
    print("\nConfusion matrix (mesh + model):")
    print(pd.DataFrame(confusion_matrix(y_true, system_pred, labels=matrix_labels),
                       index=matrix_labels, columns=matrix_labels))
    print(f"\nMean time per frame: model alone {model_ms:.0f} ms | mesh + model about {estimated_ms:.0f} ms "
          f"(mesh {mesh_ms:.0f} ms)")

    return {
        "model": name, "gate": gate_name,
        "accuracy_alone": alone["accuracy"], "accuracy_mesh_model": combined["accuracy"],
        "balanced_alone": alone["balanced_acc"], "balanced_mesh_model": combined["balanced_acc"],
        "macro_f1_alone": alone["macro_f1"], "macro_f1_mesh_model": combined["macro_f1"],
        "skip_rate": skip_rate, "unsafe_wrongly_skipped": unsafe_skipped,
        **{f"{label}_skipped": int((skipped & (y_true == label)).sum()) for label in UNSAFE_LABELS},
        "model_ms": model_ms, "mesh_ms": mesh_ms, "estimated_ms": estimated_ms,
    }


def main():
    eval_df = collect_dataset(FRAME_ROOT, LIMIT_PER_CLASS)
    if len(eval_df) == 0:
        raise RuntimeError(f"No images found under {FRAME_ROOT}. Expected folders: {LABELS}")
    print("Images:", len(eval_df))

    signals = load_signals(eval_df)
    summary = []
    for gate_name, triage in GATES.items():
        reason = apply_gate(signals, triage)
        print_gate_table(signals, reason, gate_name)
        for model_name in MODELS:
            row = evaluate_model(model_name, gate_name, signals, reason)
            if row:
                summary.append(row)
    if summary:
        pd.DataFrame(summary).to_csv(SUMMARY_CSV, index=False)
        print("\nSummary saved to:", SUMMARY_CSV)
    print("Mesh readings saved to:", SIGNALS_CSV)
    print("Note: the scheduled model call (default every 10 s) cannot act on single images, so PHONE_USAGE "
          "results here are the worst case; in a live stream that check bounds how long a phone goes unseen.")


if __name__ == "__main__":
    main()
