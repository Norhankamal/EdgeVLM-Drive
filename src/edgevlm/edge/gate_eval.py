"""Offline evaluation of the face-mesh gate on saved single-image predictions (paper, Section V-I).

Each frame takes the model's saved prediction when the gate calls the model and NORMAL when it skips.
The estimated time per frame is ``(1 - skip_rate) * model_ms + mesh_ms``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from edgevlm.constants import LABELS, UNSAFE_LABELS
from edgevlm.evaluation.metrics import compute_metrics
from edgevlm.mesh.triage import ConservativeMeshTriage, HardRuleLimits, decide_gate


def gate_calls(
    features_per_frame: Sequence[Optional[dict]],
    blurred: Sequence[bool],
    use_triage: bool,
    triage: Optional[ConservativeMeshTriage] = None,
    hard: HardRuleLimits = HardRuleLimits(),
) -> list[bool]:
    """Per-frame ``call_model`` flags for a single-image evaluation (no scheduled check: frames are isolated)."""
    return [
        decide_gate(f, blurred=bool(b), seconds_since_vlm=0.0, use_triage=use_triage, triage=triage, hard=hard).invoke
        for f, b in zip(features_per_frame, blurred)
    ]


def evaluate_gate(
    true_labels: Sequence[str],
    model_predictions: Sequence[str],
    call_model: Sequence[bool],
    model_ms: float,
    mesh_ms: float = 0.0,
) -> dict:
    """Metrics of the gated system plus the gate-specific quantities."""
    y_true = np.asarray([str(t).strip().upper() for t in true_labels])
    pred = np.asarray(model_predictions)
    call = np.asarray(call_model, dtype=bool)
    if not (len(y_true) == len(pred) == len(call)):
        raise ValueError("labels, predictions and gate decisions must have the same length")

    final = np.where(call, pred, "NORMAL")
    m = compute_metrics(y_true, final)
    unsafe = np.isin(y_true, list(UNSAFE_LABELS))
    skipped = ~call
    skip_rate = float(skipped.mean())

    return {
        "n": int(len(y_true)),
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "skip_rate": skip_rate,
        "frames_skipped": int(skipped.sum()),
        "unsafe_frames_skipped": int((skipped & unsafe).sum()),
        "unsafe_to_normal_rate": float(np.mean(final[unsafe] == "NORMAL")),
        "unsafe_class_error_rate": float(np.mean(final[unsafe] != y_true[unsafe])),
        "false_alarm_rate": float(np.mean(np.isin(final[y_true == "NORMAL"], list(UNSAFE_LABELS)))),
        "estimated_ms_per_frame": float((1.0 - skip_rate) * model_ms + mesh_ms),
        "per_class": {
            label: {
                "skipped": int((skipped & (y_true == label)).sum()),
                "support": int((y_true == label).sum()),
                "recall": m["per_class"][label]["recall"],
            }
            for label in LABELS
        },
    }
