"""The any-doubt triage is wired into the real LangGraph pipeline (reason ``mesh_triage``)."""
import time

import numpy as np
import pytest

pytest.importorskip("langgraph")

from edgevlm.mesh.features import FEATURE_NAMES, to_pipeline_landmarks  # noqa: E402
from edgevlm.pipeline import langgraph_pipeline as lp  # noqa: E402


class CountingVLM(lp.VLMRuntime):
    def __init__(self):
        self.calls = 0

    def generate_label(self, face_crop, full_crop):
        self.calls += 1
        return "NORMAL"


def _frame():
    return np.random.default_rng(1).integers(0, 255, size=(480, 640, 3), dtype=np.uint8)


def _run(features):
    vlm = CountingVLM()
    graph = lp.build_graph(vlm)
    state = lp.new_frame_state(1, _frame(), to_pipeline_landmarks(features, 0.9),
                               last_vlm_timestamp=time.time(), label_confirmed=True)
    out = graph.invoke(state)
    return out, vlm.calls


def _calm():
    f = {n: 0.0 for n in FEATURE_NAMES}
    f.update(ear_mean=0.32, yaw=1.0, mar=0.05, gaze_h=0.01, gaze_v=0.01)
    return f


def test_confident_normal_frame_is_skipped():
    out, calls = _run(_calm())
    assert out["gate_decision"] == "skip" and calls == 0 and out["label"] == "NORMAL"


def test_moderate_head_turn_passes_hard_rules_but_triage_calls_the_vlm():
    out, calls = _run({**_calm(), "yaw": 22.0})
    assert out["gate_reason"] == "mesh_triage" and out["gate_decision"] == "invoke" and calls == 1
    assert "mesh_triage" in lp.VLM_TRIGGER_REASONS


def test_legacy_landmarks_without_feature_vector_keep_the_original_behaviour():
    vlm = CountingVLM()
    graph = lp.build_graph(vlm)
    state = lp.new_frame_state(1, _frame(), {"eye_aspect_ratio": 0.3, "head_yaw_deg": 20.0, "tracking_confidence": 0.9},
                               last_vlm_timestamp=time.time(), label_confirmed=True)
    out = graph.invoke(state)
    assert out["gate_decision"] == "skip"
