"""Score private saved predictions without loading a model or importing GPU libraries.

Path matching removes only the known /home/jovyan/ prefix. It never matches on
basename, folds case, or treats matching paths as proof of matching image bytes.
"""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

LABELS = ("NORMAL", "DROWSY", "LOOKING_AWAY", "PHONE_USAGE")
PREDICTIONS = (*LABELS, "INVALID")


def normalize_path(value):
    """Normalize the recorded server prefix, not the image identity itself."""
    prefix = "/home/jovyan/"
    return value[len(prefix):] if value.startswith(prefix) else value


def _boolean(value):
    if value in ("True", "true", "1"):
        return True
    if value in ("False", "false", "0"):
        return False
    raise ValueError(f"Invalid boolean: {value!r}")


def read_predictions(path):
    """Read known export schemas; reject missing/duplicate paths and bad labels."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        true_col = "true_label" if "true_label" in fields else "label"
        pred_col = "prediction" if "prediction" in fields else "parsed_label"
        if not {"frame_path", true_col, pred_col}.issubset(fields):
            raise ValueError("CSV must contain frame_path and true/predicted label columns")
        records, indices = {}, set()
        for row in reader:
            key = normalize_path(row["frame_path"])
            if not key or key in records:
                raise ValueError("Empty or duplicate normalized frame path")
            if row[true_col] not in LABELS or row[pred_col] not in PREDICTIONS:
                raise ValueError("Unknown true or predicted label; no silent coercion")
            if "row_index" in fields:
                index = row["row_index"]
                if not index or index in indices:
                    raise ValueError("Empty or duplicate row_index")
                indices.add(index)
            record = {"true": row[true_col], "pred": row[pred_col]}
            for flag in ("exact_label_valid", "parser_failure"):
                if flag in fields:
                    record[flag] = _boolean(row[flag])
            for name in ("latency_ms", "vlm_latency_ms", "graph_latency_ms"):
                if name in fields:
                    value = float(row[name])
                    if not math.isfinite(value) or value < 0:
                        raise ValueError("Latency must be finite and nonnegative")
                    record[name] = value
            records[key] = record
    if not records:
        raise ValueError("Empty prediction CSV")
    return records


def metrics_from_matrix(matrix):
    """Four true classes, five predicted classes, zero-division P/R/F1 = 0."""
    if len(matrix) != 4 or any(len(row) != 5 for row in matrix):
        raise ValueError("Expected a 4 x 5 confusion matrix including INVALID")
    if any(type(x) is not int or x < 0 for row in matrix for x in row):
        raise ValueError("Counts must be nonnegative integers")
    supports = [sum(row) for row in matrix]
    n = sum(supports)
    if not n:
        raise ValueError("Cannot score zero predictions")
    per_class = {}
    for i, label in enumerate(LABELS):
        tp, support = matrix[i][i], supports[i]
        predicted = sum(row[i] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    correct = sum(matrix[i][i] for i in range(4))
    unsafe = sum(supports[1:])
    misses = sum(matrix[i][0] for i in range(1, 4))
    unsafe_errors = unsafe - sum(matrix[i][i] for i in range(1, 4))
    false_alarms = sum(matrix[0][1:4])  # INVALID is reported separately, not called unsafe.
    present_recalls = [per_class[label]["recall"] for i, label in enumerate(LABELS) if supports[i]]
    return {
        "images": n, "correct_predictions": correct,
        "accuracy_percent": 100 * correct / n,
        "balanced_accuracy_percent": 100 * sum(present_recalls) / len(present_recalls),
        "macro_f1_percent": 100 * sum(p["f1"] for p in per_class.values()) / 4,
        "macro_f1_definition": "Unweighted mean across the four fixed target labels",
        "unsafe_to_safe_miss_count": misses,
        "unsafe_to_safe_miss_rate_percent": 100 * misses / unsafe if unsafe else None,
        "unsafe_class_error_rate_percent": 100 * unsafe_errors / unsafe if unsafe else None,
        "false_alarm_count": false_alarms,
        "false_alarm_rate_percent": 100 * false_alarms / supports[0] if supports[0] else None,
        "invalid_prediction_count": sum(row[4] for row in matrix),
        "per_class": per_class,
        "confusion_matrix": {"true_labels": list(LABELS), "predicted_labels": list(PREDICTIONS), "matrix": matrix},
    }


def score(records):
    matrix = [[0] * 5 for _ in LABELS]
    for row in records.values():
        matrix[LABELS.index(row["true"])][PREDICTIONS.index(row["pred"])] += 1
    result = metrics_from_matrix(matrix)
    for flag in ("exact_label_valid", "parser_failure"):
        if all(flag in row for row in records.values()):
            result[flag + "_percent"] = 100 * sum(row[flag] for row in records.values()) / len(records)
    timings = {}
    for name in ("latency_ms", "vlm_latency_ms", "graph_latency_ms"):
        if not all(name in row for row in records.values()):
            continue
        values = sorted(row[name] for row in records.values())
        at = (len(values) - 1) * 0.95
        lower, upper = math.floor(at), math.ceil(at)
        p95 = values[lower] + (values[upper] - values[lower]) * (at - lower)
        timings[name] = {"mean_ms": sum(values) / len(values), "p95_ms": p95}
    result["recorded_latency"] = timings
    return result


def shared_subset(records, reference):
    """Require every reference path and label; never silently drop mismatches."""
    if not reference:
        raise ValueError("Empty reference")
    missing = set(reference) - set(records)
    if missing:
        raise ValueError(f"Missing {len(missing)} reference paths")
    if any(records[key]["true"] != row["true"] for key, row in reference.items()):
        raise ValueError("True labels differ on matched paths")
    return {key: records[key] for key in reference}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
