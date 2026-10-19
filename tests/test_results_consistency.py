"""Stored result files must be internally consistent: recomputing every metric from the stored
confusion matrix has to reproduce the stored percentages."""
import json
from pathlib import Path

import pytest

from edgevlm.constants import LABELS
from edgevlm.evaluation.metrics import compute_metrics

RESULTS = sorted((Path(__file__).resolve().parents[1] / "results" / "smolvlm_500m").glob("phase*.json"))


@pytest.mark.parametrize("path", RESULTS, ids=lambda p: p.name)
def test_metrics_follow_from_confusion_matrix(path):
    d = json.loads(path.read_text())
    cm = d["confusion_matrix"]["matrix"]
    y_true, y_pred = [], []
    for i, row in enumerate(cm):
        for j, n in enumerate(row):
            y_true += [LABELS[i]] * n
            y_pred += [(list(LABELS) + ["INVALID"])[j]] * n
    m = compute_metrics(y_true, y_pred)
    t = d["test"]
    assert m["n"] == d["data"]["test_rows"]
    for stored, computed in [("accuracy_pct", "accuracy"), ("balanced_accuracy_pct", "balanced_accuracy"),
                             ("macro_f1_pct", "macro_f1"), ("unsafe_miss_rate_pct", "unsafe_miss_rate"),
                             ("false_alarm_rate_pct", "false_alarm_rate")]:
        assert round(m[computed] * 100, 2) == pytest.approx(t[stored], abs=0.011), (path.name, stored)
    for label in LABELS:
        assert m["per_class"][label]["support"] == d["per_class"][label]["support"]
        assert round(m["per_class"][label]["recall"], 3) == pytest.approx(d["per_class"][label]["recall"], abs=0.0011)
