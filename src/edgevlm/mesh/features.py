"""The 26-entry face-mesh feature vector (paper, Section IV-B).

Entries
-------
* eye aspect ratio: mean, left, right, left-right difference            (4)
* head yaw, pitch, roll in degrees                                       (3)
* face centre x, y and face width, normalised by the image size          (3)
* mouth aspect ratio                                                     (1)
* horizontal and vertical iris gaze offsets                              (2)
* 13 blendshape scores: eye blink (L, R), eye look in/out/up/down (L, R),
  jaw open, mouth smile (L, R)                                           (13)

The geometry functions are plain NumPy and are unit-tested. ``MeshFeatureExtractor`` wraps the
MediaPipe FaceLandmarker task (478 landmarks + blendshapes) and imports MediaPipe lazily, so the
rest of the package works on machines without it.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

BLENDSHAPE_NAMES: tuple[str, ...] = (
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "jawOpen",
    "mouthSmileLeft", "mouthSmileRight",
)

FEATURE_NAMES: tuple[str, ...] = (
    "ear_mean", "ear_left", "ear_right", "ear_diff",
    "yaw", "pitch", "roll",
    "face_cx", "face_cy", "face_w",
    "mar",
    "gaze_h", "gaze_v",
) + BLENDSHAPE_NAMES

assert len(FEATURE_NAMES) == 26

# MediaPipe FaceMesh landmark indices (six points per eye, ordered p1..p6 for the EAR formula).
RIGHT_EYE_EAR = (33, 160, 158, 133, 153, 144)
LEFT_EYE_EAR = (362, 385, 387, 263, 373, 380)
RIGHT_IRIS_CENTRE, LEFT_IRIS_CENTRE = 468, 473
RIGHT_EYE_CORNERS, LEFT_EYE_CORNERS = (33, 133), (362, 263)
RIGHT_EYE_LIDS, LEFT_EYE_LIDS = (159, 145), (386, 374)
MOUTH_VERTICAL, MOUTH_HORIZONTAL = (13, 14), (78, 308)

# Six-point generic 3-D face model (nose tip, chin, eye outer corners, mouth corners), arbitrary units.
_MODEL_POINTS = np.array(
    [(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
     (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)],
    dtype=np.float64,
)
_POSE_LANDMARKS = (1, 152, 33, 263, 61, 291)


def eye_aspect_ratio(points: np.ndarray) -> float:
    """EAR from six eye landmarks p1..p6: (|p2-p6| + |p3-p5|) / (2 |p1-p4|)."""
    p = np.asarray(points, dtype=float)
    if p.shape[0] != 6:
        raise ValueError("eye_aspect_ratio needs exactly six points")
    horizontal = np.linalg.norm(p[0] - p[3])
    if horizontal == 0:
        return 0.0
    return float((np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2.0 * horizontal))


def mouth_aspect_ratio(upper: np.ndarray, lower: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    """Inner-lip opening divided by mouth width."""
    width = np.linalg.norm(np.asarray(left, float) - np.asarray(right, float))
    if width == 0:
        return 0.0
    return float(np.linalg.norm(np.asarray(upper, float) - np.asarray(lower, float)) / width)


def iris_offset(iris: np.ndarray, corners: np.ndarray, lids: np.ndarray) -> tuple[float, float]:
    """Iris centre relative to the eye centre, normalised by the eye width (horizontal, vertical)."""
    corners, lids, iris = (np.asarray(a, float) for a in (corners, lids, iris))
    width = np.linalg.norm(corners[0] - corners[1])
    if width == 0:
        return 0.0, 0.0
    centre = (corners[0] + corners[1] + lids[0] + lids[1]) / 4.0
    d = (iris[:2] - centre[:2]) / width
    return float(d[0]), float(d[1])


def head_pose_degrees(landmarks_px: np.ndarray, image_size: tuple[int, int]) -> tuple[float, float, float]:
    """Yaw, pitch, roll in degrees from a perspective-n-point fit against the six-point face model."""
    import cv2

    h, w = image_size
    image_points = np.asarray([landmarks_px[i][:2] for i in _POSE_LANDMARKS], dtype=np.float64)
    focal = float(w)
    camera = np.array([[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_MODEL_POINTS, image_points, camera, np.zeros((4, 1)),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    rot, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rot)
    pitch, yaw, roll = (float(a) for a in angles)
    # RQDecomp3x3 returns pitch near +/-180 for a frontal face under this model; fold into [-90, 90].
    pitch = (pitch + 180.0) % 360.0 - 180.0
    if abs(pitch) > 90.0:
        pitch = pitch - np.sign(pitch) * 180.0
    return yaw, pitch, roll


def features_from_landmarks(
    landmarks_px: np.ndarray,
    blendshapes: Mapping[str, float],
    image_size: tuple[int, int],
    pose: Optional[tuple[float, float, float]] = None,
) -> dict[str, float]:
    """Assemble the 26-entry feature dict from 478 pixel landmarks and the blendshape scores."""
    lm = np.asarray(landmarks_px, dtype=float)
    h, w = image_size
    ear_r = eye_aspect_ratio(lm[list(RIGHT_EYE_EAR)][:, :2])
    ear_l = eye_aspect_ratio(lm[list(LEFT_EYE_EAR)][:, :2])
    yaw, pitch, roll = pose if pose is not None else head_pose_degrees(lm, image_size)

    xs, ys = lm[:, 0], lm[:, 1]
    gh_r, gv_r = iris_offset(lm[RIGHT_IRIS_CENTRE], lm[list(RIGHT_EYE_CORNERS)], lm[list(RIGHT_EYE_LIDS)])
    gh_l, gv_l = iris_offset(lm[LEFT_IRIS_CENTRE], lm[list(LEFT_EYE_CORNERS)], lm[list(LEFT_EYE_LIDS)])

    feats: dict[str, float] = {
        "ear_mean": (ear_l + ear_r) / 2.0, "ear_left": ear_l, "ear_right": ear_r, "ear_diff": ear_l - ear_r,
        "yaw": yaw, "pitch": pitch, "roll": roll,
        "face_cx": float((xs.min() + xs.max()) / 2.0 / w),
        "face_cy": float((ys.min() + ys.max()) / 2.0 / h),
        "face_w": float((xs.max() - xs.min()) / w),
        "mar": mouth_aspect_ratio(lm[MOUTH_VERTICAL[0]], lm[MOUTH_VERTICAL[1]],
                                  lm[MOUTH_HORIZONTAL[0]], lm[MOUTH_HORIZONTAL[1]]),
        "gaze_h": (gh_l + gh_r) / 2.0, "gaze_v": (gv_l + gv_r) / 2.0,
    }
    for name in BLENDSHAPE_NAMES:
        feats[name] = float(blendshapes.get(name, 0.0))
    return feats


def to_pipeline_landmarks(features: Optional[Mapping[str, float]], tracking_confidence: float = 1.0) -> Optional[dict]:
    """Convert a feature dict to the ``landmarks`` mapping the LangGraph pipeline reads."""
    if features is None:
        return None
    return {
        "eye_aspect_ratio": features["ear_mean"],
        "head_yaw_deg": features["yaw"],
        "tracking_confidence": tracking_confidence,
        "features": dict(features),
    }


class MeshFeatureExtractor:
    """MediaPipe FaceLandmarker wrapper returning the 26-entry feature dict (or ``None`` without a face)."""

    def __init__(self, model_path: str):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            output_face_blendshapes=True,
            num_faces=1,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def extract(self, rgb_frame: np.ndarray) -> Optional[dict[str, float]]:
        h, w = rgb_frame.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))
        result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return None
        pts = np.array([(p.x * w, p.y * h, p.z * w) for p in result.face_landmarks[0]], dtype=float)
        shapes: Sequence = result.face_blendshapes[0] if result.face_blendshapes else []
        return features_from_landmarks(pts, {b.category_name: b.score for b in shapes}, (h, w))
