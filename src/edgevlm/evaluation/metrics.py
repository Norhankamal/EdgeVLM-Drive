"""Output parsing and evaluation metrics, with the exact definitions used in the notebooks.

* ``parse_prediction`` - a response counts as a valid label if, after stripping whitespace
  and ``.,;:!?'"``, it equals one of the four labels; it is *exactly* valid only if it
  already equals the label with no punctuation. Anything else is ``INVALID``.
* **Unsafe-state miss rate** - fraction of truly unsafe frames (DROWSY / LOOKING_AWAY /
  PHONE_USAGE) whose prediction differs from their true label (so confusing two unsafe
  states counts as a miss, as in the report).
* **False-alarm rate** - fraction of NORMAL frames predicted as any unsafe state.
* **Balanced accuracy** - mean per-class recall over the four classes.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from edgevlm.constants import INVALID_LABEL, LABELS, UNSAFE_LABELS

_STRIP_CHARS = " .,;:!?'\""


def parse_prediction(raw_text: str, labels: Sequence[str] = LABELS) -> tuple[str, bool]:
    """Map raw model text to ``(label, exactly_valid)``; unmappable text -> ``("INVALID", False)``."""
    normalized = str(raw_text).strip().upper()
    cleaned = normalized.strip(_STRIP_CHARS)
    if cleaned in labels:
        return cleaned, (normalized in labels)
    return INVALID_LABEL, False


def _rate(flags) -> float:
    """Mean of a boolean array; NaN (without a numpy warning) when the group is empty."""
    return float(np.mean(flags)) if len(flags) else float("nan")


def compute_metrics(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    exact_valid_flags: Optional[Sequence[bool]] = None,
    latencies_ms: Optional[Sequence[float]] = None,
    labels: Sequence[str] = LABELS,
) -> dict:
    """Return the full metric dictionary reported by the evaluation notebooks (values as fractions)."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    labels = list(labels)
    y_true = np.asarray([str(x).strip().upper() for x in true_labels])
    y_pred = np.asarray(list(predicted_labels))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    unsafe_mask = np.isin(y_true, list(UNSAFE_LABELS))
    normal_mask = y_true == "NORMAL"

    matrix_labels = labels + [INVALID_LABEL]
    metrics: dict = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "unsafe_miss_rate": _rate(y_pred[unsafe_mask] != y_true[unsafe_mask]),
        "false_alarm_rate": _rate(np.isin(y_pred[normal_mask], list(UNSAFE_LABELS))),
        "parser_failure_rate": float(np.mean(y_pred == INVALID_LABEL)),
        "per_class": {
            label: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for label, p, r, f, s in zip(labels, precision, recall, f1, support)
        },
        "confusion_matrix": {
            "labels": matrix_labels,
            "rows_true_cols_pred": confusion_matrix(y_true, y_pred, labels=matrix_labels).tolist(),
        },
    }
    if exact_valid_flags is not None:
        metrics["exact_label_validity_rate"] = float(np.mean(np.asarray(exact_valid_flags, dtype=bool)))
    if latencies_ms is not None and len(latencies_ms):
        lat = np.asarray(latencies_ms, dtype=float)
        metrics["latency_ms"] = {"mean": float(lat.mean()), "p95": float(np.percentile(lat, 95))}
    return metrics


def format_report(metrics: dict, title: str = "TEST RESULTS") -> str:
    """Human-readable block in the same layout the notebooks print."""
    lines = [f"========== {title} ==========", f"Evaluation images: {metrics['n']}"]
    if "exact_label_validity_rate" in metrics:
        lines.append(f"Exact-label validity rate: {metrics['exact_label_validity_rate'] * 100:.2f} %")
    lines += [
        f"Parser failure rate: {metrics['parser_failure_rate'] * 100:.2f} %",
        f"Overall accuracy: {metrics['accuracy'] * 100:.2f} %",
        f"Balanced accuracy: {metrics['balanced_accuracy'] * 100:.2f} %",
        f"Macro-F1: {metrics['macro_f1'] * 100:.2f} %",
        "",
        "Per-class results:",
    ]
    for label, m in metrics["per_class"].items():
        lines.append(
            f"{label}: precision={m['precision']:.3f}, recall={m['recall']:.3f}, "
            f"F1={m['f1']:.3f}, samples={m['support']}"
        )
    lines += [
        "",
        f"Unsafe-state miss rate: {metrics['unsafe_miss_rate'] * 100:.2f} %",
        f"False-alarm rate: {metrics['false_alarm_rate'] * 100:.2f} %",
    ]
    if "latency_ms" in metrics:
        lines += [
            f"Mean latency (ms): {metrics['latency_ms']['mean']:.2f}",
            f"P95 latency (ms): {metrics['latency_ms']['p95']:.2f}",
        ]
    return "\n".join(lines)
