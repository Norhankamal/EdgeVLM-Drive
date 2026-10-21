import pytest

from edgevlm.edge.gate_eval import evaluate_gate
from edgevlm.edge.stats import mcnemar_exact, paired_counts, wilson_interval


def test_gate_eval_arithmetic():
    true = ["NORMAL", "DROWSY", "PHONE_USAGE", "PHONE_USAGE"]
    pred = ["NORMAL", "DROWSY", "PHONE_USAGE", "PHONE_USAGE"]
    call = [True, True, False, True]            # one phone frame skipped -> counted NORMAL
    r = evaluate_gate(true, pred, call, model_ms=1000.0, mesh_ms=100.0)
    assert r["accuracy"] == pytest.approx(0.75)
    assert r["frames_skipped"] == 1 and r["unsafe_frames_skipped"] == 1
    assert r["unsafe_to_normal_rate"] == pytest.approx(1 / 3)
    assert r["false_alarm_rate"] == 0.0
    assert r["estimated_ms_per_frame"] == pytest.approx(0.75 * 1000 + 100)
    assert r["per_class"]["PHONE_USAGE"]["skipped"] == 1


def test_gate_eval_rejects_length_mismatch():
    with pytest.raises(ValueError):
        evaluate_gate(["NORMAL"], ["NORMAL", "NORMAL"], [True], 1.0)


def test_wilson_and_mcnemar_match_reported_values():
    lo, hi = wilson_interval(994, 1200)
    assert (round(100 * lo, 1), round(100 * hi, 1)) == (80.6, 84.9)
    assert mcnemar_exact(82, 37) == pytest.approx(4.5e-5, rel=0.01)
    assert mcnemar_exact(0, 0) == 1.0
    assert paired_counts([1, 1, 0, 0], [1, 0, 1, 0]) == {
        "both_correct": 1, "only_a_correct": 1, "only_b_correct": 1, "both_wrong": 1}
