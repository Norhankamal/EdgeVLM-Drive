"""The stored on-device results must be internally consistent: every derived number is recomputed."""
import json
from pathlib import Path

import numpy as np
import pytest

from edgevlm.edge.stats import mcnemar_exact, wilson_interval

R = Path(__file__).resolve().parents[1] / "results" / "jetson"


def load(name):
    return json.loads((R / name).read_text())


def models():
    for f in ("phase1_validation.json", "phase2_balanced_1200.json"):
        d = load(f)
        for name, m in d["models"].items():
            yield d, name, m


@pytest.mark.parametrize("d,name,m", list(models()), ids=lambda x: x if isinstance(x, str) else "")
def test_metrics_follow_from_confusion_matrix(d, name, m):
    cm = np.array(m["confusion_matrix"]["rows_true_cols_pred"])
    assert cm.sum() == d["images"] == m["n"]
    assert (cm.sum(1) == d["per_class"]).all()
    assert m["accuracy_pct"] == pytest.approx(100 * np.trace(cm) / cm.sum(), abs=0.006)
    unsafe = cm[1:].sum()
    assert m["unsafe_to_normal_rate_pct"] == pytest.approx(100 * cm[1:, 0].sum() / unsafe, abs=0.006)
    assert m["false_alarm_rate_pct"] == pytest.approx(100 * cm[0, 1:].sum() / cm[0].sum(), abs=0.006)
    lo, hi = wilson_interval(int(np.trace(cm)), int(cm.sum()))
    assert m["accuracy_ci95_pct"] == [round(100 * lo, 1), round(100 * hi, 1)]
    for i, label in enumerate(m["confusion_matrix"]["labels"]):
        assert m["per_class"][label]["recall"] == pytest.approx(cm[i, i] / cm[i].sum(), abs=0.001)


def test_paired_comparison():
    p = load("phase2_balanced_1200.json")
    c = p["paired_comparison"]
    assert sum(c[k] for k in ("both_correct", "only_256m_correct", "only_500m_correct", "both_wrong")) == 1200
    m = p["models"]
    assert c["both_correct"] + c["only_256m_correct"] == m["SmolVLM-256M_Phase2_V2"]["correct"]
    assert c["both_correct"] + c["only_500m_correct"] == m["SmolVLM-500M_Phase2_V1"]["correct"]
    assert c["mcnemar_exact_p"] == pytest.approx(mcnemar_exact(82, 37))
    assert c["mcnemar_exact_p"] < 0.001


def test_latency_and_throughput_are_consistent():
    p = load("phase2_balanced_1200.json")
    for m in p["models"].values():
        assert m["throughput_frames_per_min"] == pytest.approx(60000 / m["latency_ms"]["mean"], abs=0.05)
    m = p["models"]
    ratio = 1 - m["SmolVLM-256M_Phase2_V2"]["peak_gpu_memory_mb"] / m["SmolVLM-500M_Phase2_V1"]["peak_gpu_memory_mb"]
    assert 100 * ratio == pytest.approx(40, abs=1)  # "about 40% less GPU memory"


def test_gate_table_is_consistent():
    g = load("gate_experiments.json")
    n = g["images"]
    for rule, skipped in g["skipped_frames_by_class"].items():
        total = sum(skipped.values())
        for r in g["results"]:
            if r["configuration"].replace("mesh, ", "").replace(" + ", "_plus_").replace(" ", "_") == rule:
                assert r["frames_skipped_pct"] == pytest.approx(100 * total / n, abs=0.06)
                assert r["unsafe_frames_skipped"] == total - skipped["NORMAL"]
    for r in g["results"]:  # accuracy equals the recall-weighted mean over the four balanced classes
        assert r["accuracy_pct"] == pytest.approx(100 * sum(r["recall"].values()) / 4, abs=0.15)
        if r["configuration"] != "model only":
            base = next(x for x in g["results"] if x["model"] == r["model"] and x["configuration"] == "model only")
            skip = r["frames_skipped_pct"] / 100
            expected = (1 - skip) * base["estimated_ms_per_frame"] + g["mesh"]["mean_ms_per_frame"]
            assert r["estimated_ms_per_frame"] == pytest.approx(expected, abs=4)
