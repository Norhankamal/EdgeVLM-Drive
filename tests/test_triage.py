import pytest

from edgevlm.mesh.features import BLENDSHAPE_NAMES, FEATURE_NAMES
from edgevlm.mesh.triage import ConservativeMeshTriage, TriageLimits, decide_gate, hard_rule_reasons


def attentive():
    """A clearly attentive, frontal face: every triage condition satisfied."""
    f = {name: 0.0 for name in FEATURE_NAMES}
    f.update(ear_mean=0.32, ear_left=0.32, ear_right=0.32, yaw=2.0, pitch=0.0, roll=1.0, mar=0.05,
             gaze_h=0.02, gaze_v=0.03)
    return f


def test_feature_vector_has_26_entries_and_13_blendshapes():
    assert len(FEATURE_NAMES) == 26 and len(set(FEATURE_NAMES)) == 26
    assert len(BLENDSHAPE_NAMES) == 13


def test_attentive_frame_is_confidently_normal():
    assert ConservativeMeshTriage().doubts(attentive()) == []


@pytest.mark.parametrize(
    "change,doubt",
    [
        ({"ear_mean": 0.24}, "ear"),          # eyes not fully open
        ({"yaw": 20.0}, "yaw"),               # moderate head turn that the hard rule (30 deg) misses
        ({"roll": -30.0}, "roll"),
        ({"gaze_h": 0.2}, "gaze_h"),
        ({"gaze_v": -0.2}, "gaze_v"),
        ({"mar": 0.5}, "mar"),                # mouth open
        ({"eyeLookDownLeft": 0.5}, "look_down"),
        ({"eyeLookOutRight": 0.6}, "look_other"),
        ({"eyeBlinkLeft": 0.6}, "blink"),
    ],
)
def test_any_single_doubtful_reading_sends_frame_to_vlm(change, doubt):
    f = attentive()
    f.update(change)
    assert doubt in ConservativeMeshTriage().doubts(f)


def test_missing_reading_counts_as_doubt():
    f = attentive()
    del f["gaze_h"]
    assert "gaze_h" in ConservativeMeshTriage().doubts(f)
    assert ConservativeMeshTriage().doubts(None) == ["no_features"]


def test_pitch_limit_disabled_by_default_but_configurable():
    f = attentive()
    f["pitch"] = 60.0
    assert ConservativeMeshTriage().confident_normal(f)
    assert "pitch" in ConservativeMeshTriage(TriageLimits(pitch_max_deg=25.0)).doubts(f)


def test_hard_rules():
    assert hard_rule_reasons(None) == ["no_face"]
    f = attentive()
    assert hard_rule_reasons(f) == []
    assert hard_rule_reasons({**f, "ear_mean": 0.15}) == ["eyes_closed"]
    assert hard_rule_reasons({**f, "yaw": 35.0}) == ["head_turned"]
    assert hard_rule_reasons(f, blurred=True) == ["poor_frame_quality"]


def test_gate_order_and_reasons():
    f = attentive()
    assert decide_gate(f).invoke is False
    assert decide_gate({**f, "yaw": 20.0}).reason == "mesh_triage"
    assert decide_gate({**f, "yaw": 20.0}, use_triage=False).invoke is False   # rules-only gate skips it
    assert decide_gate({**f, "ear_mean": 0.1}).reason == "eyes_closed"          # hard rule wins over triage
    assert decide_gate(f, seconds_since_vlm=11.0).reason == "scheduled_check"
    assert decide_gate(f, seconds_since_vlm=9.0).invoke is False
