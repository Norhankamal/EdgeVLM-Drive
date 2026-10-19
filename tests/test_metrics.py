"""The metric code must reproduce the numbers printed by the executed notebooks and reported
in the interim report, starting from the published confusion matrices."""
import numpy as np
import pytest

from edgevlm.constants import LABELS
from edgevlm.evaluation.metrics import compute_metrics, parse_prediction

# Phase 1 test confusion matrix (interim report, Table VIII); rows = true, cols = predicted.
PHASE1_CM = [[2014, 44, 73, 20], [73, 90, 3, 2], [59, 3, 237, 3], [145, 30, 77, 706]]


def expand(cm):
    y_true, y_pred = [], []
    for i, row in enumerate(cm):
        for j, n in enumerate(row):
            y_true += [LABELS[i]] * n
            y_pred += [LABELS[j]] * n
    return y_true, y_pred


def test_phase1_numbers_match_report():
    m = compute_metrics(*expand(PHASE1_CM))
    assert m["n"] == 3579
    assert round(m["accuracy"] * 100, 2) == 85.14
    assert round(m["balanced_accuracy"] * 100, 2) == 74.84
    assert round(m["macro_f1"] * 100, 2) == 74.13
    assert round(m["unsafe_miss_rate"] * 100, 2) == 27.66
    assert round(m["false_alarm_rate"] * 100, 2) == 6.37
    assert round(m["per_class"]["DROWSY"]["recall"], 3) == 0.536
    assert m["per_class"]["DROWSY"]["support"] == 168


def test_phase2_final_numbers_match_notebook():
    cm = [[3023, 186, 31, 37], [186, 826, 5, 4], [84, 2, 215, 1], [117, 5, 108, 728]]
    m = compute_metrics(*expand(cm))
    assert m["n"] == 5558
    assert round(m["accuracy"] * 100, 2) == 86.22
    assert round(m["balanced_accuracy"] * 100, 2) == 80.08
    assert round(m["macro_f1"] * 100, 2) == 80.18
    assert round(m["unsafe_miss_rate"] * 100, 2) == 22.45
    assert round(m["false_alarm_rate"] * 100, 2) == 7.75


def test_invalid_predictions_counted():
    m = compute_metrics(["NORMAL", "DROWSY"], ["INVALID", "DROWSY"])
    assert m["parser_failure_rate"] == 0.5
    assert m["confusion_matrix"]["labels"][-1] == "INVALID"
    assert m["confusion_matrix"]["rows_true_cols_pred"][0][4] == 1


def test_latency_percentiles():
    m = compute_metrics(["NORMAL"] * 100, ["NORMAL"] * 100, latencies_ms=list(range(1, 101)))
    assert m["latency_ms"]["mean"] == pytest.approx(50.5)
    assert m["latency_ms"]["p95"] == pytest.approx(np.percentile(range(1, 101), 95))


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("NORMAL", ("NORMAL", True)),
        ("  drowsy ", ("DROWSY", True)),          # case/whitespace normalised -> exactly valid
        ("NORMAL.", ("NORMAL", False)),           # punctuation: usable label, not an exact match
        ("'LOOKING_AWAY'!", ("LOOKING_AWAY", False)),
        ("The driver is drowsy", ("INVALID", False)),
        ("", ("INVALID", False)),
    ],
)
def test_parse_prediction(raw, expected):
    assert parse_prediction(raw) == expected
