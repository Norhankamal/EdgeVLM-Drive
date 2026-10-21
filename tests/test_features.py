import numpy as np
import pytest

from edgevlm.mesh.features import (
    FEATURE_NAMES,
    eye_aspect_ratio,
    features_from_landmarks,
    iris_offset,
    mouth_aspect_ratio,
    to_pipeline_landmarks,
)


def _eye(opening):
    """Six eye points p1..p6 for a 30-px-wide eye with the given vertical opening."""
    o = opening / 2
    return np.array([(0, 0), (10, -o), (20, -o), (30, 0), (20, o), (10, o)], float)


def test_ear_open_vs_closed():
    assert eye_aspect_ratio(_eye(10)) == pytest.approx(10 / 30)
    assert eye_aspect_ratio(_eye(0)) == 0.0
    assert eye_aspect_ratio(_eye(10)) > 0.25 > eye_aspect_ratio(_eye(5))


def test_ear_needs_six_points():
    with pytest.raises(ValueError):
        eye_aspect_ratio(np.zeros((5, 2)))


def test_mouth_and_iris_geometry():
    assert mouth_aspect_ratio((0, 0), (0, 6), (-10, 3), (10, 3)) == pytest.approx(0.3)
    corners, lids = np.array([[0.0, 5.0], [30.0, 5.0]]), np.array([[15.0, 0.0], [15.0, 10.0]])
    h, v = iris_offset(np.array([15.0, 5.0]), corners, lids)
    assert (h, v) == (0.0, 0.0)
    h, _ = iris_offset(np.array([21.0, 5.0]), corners, lids)
    assert h == pytest.approx(0.2)


def test_feature_dict_has_every_entry_and_pipeline_mapping():
    rng = np.random.default_rng(0)
    lm = rng.uniform(100, 500, size=(478, 3))
    feats = features_from_landmarks(lm, {"jawOpen": 0.2, "eyeBlinkLeft": 0.1}, (640, 640), pose=(3.0, -2.0, 1.0))
    assert set(feats) == set(FEATURE_NAMES)
    assert feats["jawOpen"] == 0.2 and feats["mouthSmileLeft"] == 0.0 and feats["yaw"] == 3.0
    assert feats["ear_diff"] == pytest.approx(feats["ear_left"] - feats["ear_right"])
    pl = to_pipeline_landmarks(feats, 0.9)
    assert pl["eye_aspect_ratio"] == feats["ear_mean"] and pl["head_yaw_deg"] == 3.0 and pl["features"] == feats
    assert to_pipeline_landmarks(None) is None


def test_head_pose_recovers_yaw_from_a_synthetic_projection():
    cv2 = pytest.importorskip("cv2")
    from edgevlm.mesh import features as F

    yaw_true = 25.0
    rvec = np.array([[0.0], [np.deg2rad(yaw_true)], [0.0]])
    tvec = np.array([[0.0], [0.0], [1200.0]])
    w = h = 640
    cam = np.array([[w, 0, w / 2], [0, h, h / 2], [0, 0, 1]], float)
    proj, _ = cv2.projectPoints(F._MODEL_POINTS, rvec, tvec, cam, np.zeros((4, 1)))
    lm = np.zeros((478, 2))
    for idx, p in zip(F._POSE_LANDMARKS, proj.reshape(-1, 2)):
        lm[idx] = p
    yaw, pitch, roll = F.head_pose_degrees(lm, (h, w))
    assert abs(abs(yaw) - yaw_true) < 3.0 and abs(pitch) < 5.0 and abs(roll) < 5.0
