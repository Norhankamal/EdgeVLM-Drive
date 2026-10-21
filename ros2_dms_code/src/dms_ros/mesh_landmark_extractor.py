"""
Driver Monitoring - MediaPipe Face Mesh landmark extraction
=============================================================

Owns the one piece `langgraph_pipeline.py` explicitly does NOT own (see that
module's top-level docstring: "This module ... deliberately does not own
camera I/O, MediaPipe, or the VLM runtime - those live in their own ROS
nodes / modules and are called through the small interfaces below"). This
file IS that separate module for the MediaPipe side: it turns a raw camera
frame into exactly the `landmarks` dict `assess_risk()` / `MeshInferenceAdapter`
in `langgraph_pipeline.py` already expect:

    {
        "eye_aspect_ratio": float,   # lower = eyes more closed
        "head_yaw_deg": float,       # 0 = facing camera, +/- = turned away
        "tracking_confidence": float,  # see the note on this field below
    }

or `None` when no face is found at all (the pipeline already treats `None`
as "tracking uncertain" and forces a VLM call - see `assess_risk()`'s
docstring in `langgraph_pipeline.py`).

Nothing in `langgraph_pipeline.py` needed to change for this to plug in -
`mesh_signals_node`, `assess_risk()`, and `MeshInferenceAdapter` were
already written against this exact contract and were just waiting for a
real producer of `landmarks` instead of the `None`/hand-built dicts the
unit tests and demo scenarios use.

Uses the MediaPipe Tasks API (`FaceLandmarker`), not the older
`mp.solutions.face_mesh` - the Tasks API is MediaPipe's current, maintained
face-landmark interface and is what new integrations should target.

--- Jetson Nano / aarch64 installation note (read this before deploying) ---
The official `mediapipe` PyPI wheels do not reliably cover Jetson's aarch64
+ older JetPack/CUDA combination out of the box. Before running this on the
Jetson, confirm `python3 -c "import mediapipe"` actually imports there; if
it does not, use a community-built aarch64 wheel (e.g. PINTO0309's
mediapipe-bin project: https://github.com/PINTO0309/mediapipe-bin) or build
from source for the board's specific JetPack version. This is a real
hardware-bring-up step, not a formality - budget time for it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Landmark indices
# ---------------------------------------------------------------------------
# The Tasks API FaceLandmarker uses the same 478-point mesh topology as the
# older solutions.face_mesh, so these are the standard, widely-used index
# sets for this topology (not something specific to this project).

# 6-point eye contour per eye, in the (p1, p2, p3, p4, p5, p6) order the
# classic Soukupova & Cech EAR formula expects: p1/p4 are the horizontal
# corners, (p2, p3) the upper lid, (p5, p6) the lower lid.
RIGHT_EYE_IDX = (33, 160, 158, 133, 153, 144)
LEFT_EYE_IDX = (362, 385, 387, 263, 373, 380)

# 6 points used for solvePnP head-pose: nose tip, chin, outer corner of each
# eye, and each mouth corner - the standard minimal landmark set for this
# problem (see e.g. the widely-used OpenCV head-pose-estimation tutorials
# this is adapted from).
_POSE_LANDMARK_IDX = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 263,
    "right_eye_outer": 33,
    "mouth_left": 291,
    "mouth_right": 61,
}

# Generic 3D face model, in an arbitrary millimeter-like unit, centered
# roughly at the nose tip - NOT a measurement of any specific person. This
# is the standard generic model used across OpenCV head-pose tutorials; it
# is accurate enough to get a usable yaw angle out of solvePnP without
# needing a per-driver 3D scan, but it is an approximation, not a
# calibrated fit. If the yaw numbers look biased during calibration (see
# `--calibrate` in run_live_pipeline.py), that is the first thing to
# revisit.
_MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),        # nose tip
    (0.0, -330.0, -65.0),   # chin
    (-225.0, 170.0, -135.0),  # left eye outer corner
    (225.0, 170.0, -135.0),   # right eye outer corner
    (-150.0, -150.0, -125.0),  # mouth left corner
    (150.0, -150.0, -125.0),   # mouth right corner
], dtype=np.float64)

# ---------------------------------------------------------------------------
# Extended feature set (consumed by mesh_triage.py)
# ---------------------------------------------------------------------------
# Eye/mouth/iris landmark indices of the 478-point mesh (468-477 are the iris).
_IRIS_RIGHT_CENTER, _IRIS_LEFT_CENTER = 468, 473
_RIGHT_EYE_CORNERS = (133, 33)    # inner, outer
_LEFT_EYE_CORNERS = (362, 263)    # inner, outer
_RIGHT_EYE_LIDS = (159, 145)      # upper, lower
_LEFT_EYE_LIDS = (386, 374)
_MOUTH_VERTICAL = (13, 14)
_MOUTH_HORIZONTAL = (78, 308)

# Blendshape scores kept as features (FaceLandmarker output_face_blendshapes).
BLENDSHAPE_FEATURES = (
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "jawOpen", "mouthSmileLeft", "mouthSmileRight",
)

# Fixed order of the feature vector. Every name is also a column of the
# signals CSV written by the evaluation tools, so training, offline replay
# and live inference all use the same definition.
FEATURE_NAMES = (
    "ear", "ear_left", "ear_right", "ear_diff", "yaw", "pitch", "roll",
    "face_cx", "face_cy", "face_w", "mar", "gaze_h", "gaze_v",
) + tuple("bs_" + n for n in BLENDSHAPE_FEATURES)


def wrap_pm90(angle_deg: float) -> float:
    """Fold an angle into [-90, 90). solvePnP with the generic face model can
    report head pitch/roll offset by 180 degrees; folding keeps 'facing the
    camera' near zero regardless of that branch."""
    return ((float(angle_deg) + 90.0) % 180.0) - 90.0


def _iris_gaze(points_2d: np.ndarray) -> tuple:
    """Horizontal/vertical iris position inside the eye opening, averaged over
    both eyes (0 = centred). Needs the 478-point mesh; returns zeros otherwise."""
    if len(points_2d) < 478:
        return 0.0, 0.0
    gh, gv = [], []
    for iris, (inner, outer), (up, low) in (
        (_IRIS_RIGHT_CENTER, _RIGHT_EYE_CORNERS, _RIGHT_EYE_LIDS),
        (_IRIS_LEFT_CENTER, _LEFT_EYE_CORNERS, _LEFT_EYE_LIDS),
    ):
        width = points_2d[outer][0] - points_2d[inner][0]
        height = points_2d[low][1] - points_2d[up][1]
        if abs(width) > 1e-6:
            gh.append((points_2d[iris][0] - points_2d[inner][0]) / width - 0.5)
        if abs(height) > 1e-6:
            gv.append((points_2d[iris][1] - points_2d[up][1]) / height - 0.5)
    return (float(np.mean(gh)) if gh else 0.0), (float(np.mean(gv)) if gv else 0.0)


def build_features(points_2d: np.ndarray, frame_shape: tuple, left_ear: float, right_ear: float,
                   yaw: float, pitch: float, roll: float, blendshapes: Optional[dict]) -> dict:
    """Scale-invariant per-frame features describing eyes, gaze, head pose,
    face position in the frame and mouth. Everything here is computed from the
    face mesh alone (no pixels)."""
    height, width = frame_shape[:2]
    xs, ys = points_2d[:, 0], points_2d[:, 1]
    mouth_v = np.linalg.norm(points_2d[_MOUTH_VERTICAL[0]] - points_2d[_MOUTH_VERTICAL[1]])
    mouth_h = np.linalg.norm(points_2d[_MOUTH_HORIZONTAL[0]] - points_2d[_MOUTH_HORIZONTAL[1]])
    gaze_h, gaze_v = _iris_gaze(points_2d)
    feats = {
        "ear": (left_ear + right_ear) / 2.0, "ear_left": left_ear, "ear_right": right_ear,
        "ear_diff": abs(left_ear - right_ear),
        "yaw": yaw, "pitch": wrap_pm90(pitch), "roll": wrap_pm90(roll),
        "face_cx": float(xs.mean() / width), "face_cy": float(ys.mean() / height),
        "face_w": float((xs.max() - xs.min()) / width),
        "mar": float(mouth_v / mouth_h) if mouth_h > 1e-6 else 0.0,
        "gaze_h": gaze_h, "gaze_v": gaze_v,
    }
    blendshapes = blendshapes or {}
    for name in BLENDSHAPE_FEATURES:
        feats["bs_" + name] = float(blendshapes.get(name, 0.0))
    return feats


# Default pretrained landmark model asset. MediaPipe does not bundle model
# weights inside the pip package - this `.task` file has to be downloaded
# once per machine. See `download_face_landmarker_model()` below.
DEFAULT_MODEL_PATH = Path.home() / ".cache" / "mediapipe" / "face_landmarker.task"
MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def download_face_landmarker_model(dest: Path = DEFAULT_MODEL_PATH) -> Path:
    """Fetch the pretrained `.task` model file once, if not already cached.

    Run this on a machine with internet access before relying on it on the
    Jetson - the Jetson's network access during a demo/test session should
    not be a dependency of the driver-monitoring pipeline itself. Copy the
    resulting file to the same path on the Jetson (or point
    `MeshLandmarkExtractor(model_path=...)` at wherever it was copied to).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        return dest
    import urllib.request
    urllib.request.urlretrieve(MODEL_DOWNLOAD_URL, str(dest))
    return dest


@dataclass
class LandmarkExtractionResult:
    """Return type of `MeshLandmarkExtractor.extract()` before it gets
    reduced to the plain dict `langgraph_pipeline.py` expects - kept as a
    separate, richer type so callers that want more than the three
    contract fields (e.g. the calibration tool, which also wants the raw
    EAR per eye for display) are not forced to parse a plain dict.
    """
    eye_aspect_ratio: Optional[float]
    head_yaw_deg: Optional[float]
    head_pitch_deg: Optional[float]
    tracking_confidence: Optional[float]
    left_ear: Optional[float] = None
    right_ear: Optional[float] = None
    head_roll_deg: Optional[float] = None
    features: Optional[dict] = None     # see FEATURE_NAMES

    def to_landmarks_dict(self) -> Optional[dict]:
        """The exact shape `assess_risk()` / `MeshInferenceAdapter` in
        `langgraph_pipeline.py` read - see that module's `GraphState.landmarks`
        field and `assess_risk()`'s docstring for the contract this mirrors.
        """
        if self.eye_aspect_ratio is None or self.head_yaw_deg is None:
            return None
        landmarks = {
            "eye_aspect_ratio": self.eye_aspect_ratio,
            "head_yaw_deg": self.head_yaw_deg,
            "tracking_confidence": self.tracking_confidence,
        }
        if self.features is not None:
            landmarks["mesh_features"] = self.features
        return landmarks


def _eye_aspect_ratio(points_2d: np.ndarray, idx: tuple) -> float:
    """Standard 6-point EAR: (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||).

    `points_2d` is in pixel coordinates (not normalized 0-1) so the ratio is
    scale-invariant regardless of frame resolution or how far the driver
    sits from the camera - only the *ratio* of vertical to horizontal eye
    opening matters, not absolute distances.
    """
    p1, p2, p3, p4, p5, p6 = (points_2d[i] for i in idx)
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return 0.0
    return float(vertical / (2.0 * horizontal))


def _head_pose(points_2d: np.ndarray, frame_shape: tuple) -> tuple:
    """solvePnP-based head yaw/pitch/roll in degrees, using the generic 3D model
    above and an approximate camera matrix.

    No calibration file is assumed, so the camera matrix is approximated
    from frame dimensions (focal length ~= frame width, principal point at
    the frame center) - a standard, widely-used approximation when a real
    calibration is not available. This is good enough to threshold against
    (e.g. "is the driver's head turned more than ~30 degrees"), but is not
    a precision instrument; a real camera calibration (checkerboard +
    `cv2.calibrateCamera`) would tighten this if the thresholds ever need
    to be more exact than a coarse "turned away or not" cue.
    """
    import cv2

    height, width = frame_shape[:2]
    focal_length = float(width)
    center = (width / 2.0, height / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))  # assume no lens distortion - no calibration available

    image_points = np.array([
        points_2d[_POSE_LANDMARK_IDX["nose_tip"]],
        points_2d[_POSE_LANDMARK_IDX["chin"]],
        points_2d[_POSE_LANDMARK_IDX["left_eye_outer"]],
        points_2d[_POSE_LANDMARK_IDX["right_eye_outer"]],
        points_2d[_POSE_LANDMARK_IDX["mouth_left"]],
        points_2d[_POSE_LANDMARK_IDX["mouth_right"]],
    ], dtype=np.float64)

    success, rotation_vec, _translation_vec = cv2.solvePnP(
        _MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None, None, None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
    # Rotation-matrix-to-Euler-angle decomposition, with the axis-to-angle
    # mapping verified empirically against this specific 3D model's axes
    # (X = left/right across the face, Y = up/down, Z = depth) rather than
    # assumed from a generic tutorial formula: a synthetic pure-Y-axis
    # rotation (a left/right head turn in this model) must recover as pure
    # yaw with no pitch/roll leakage, and a pure-X-axis rotation (a nod)
    # must recover as pure pitch - both were checked directly before this
    # shipped, because it is easy to silently swap yaw and pitch here (an
    # earlier version of this function did exactly that: using
    # atan2(R[1,0], R[0,0]) for yaw gave a near-constant ~0 reading
    # regardless of actual head turn, which would have made every
    # LOOKING_AWAY risk cue silently unreachable).
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    else:
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])

    roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) if not singular else 0.0
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


class MeshLandmarkExtractor:
    """Wraps MediaPipe's `FaceLandmarker` task and reduces its output to the
    `eye_aspect_ratio` / `head_yaw_deg` / `tracking_confidence` contract
    `langgraph_pipeline.py` already expects.

    `tracking_confidence` note: the Tasks API's `FaceLandmarkerResult` does
    not expose a single scalar "how confident is this reading" number the
    way a detector's output does (checked directly against the installed
    `mediapipe` package - `FaceLandmarkerResult` only carries
    `face_landmarks` / `face_blendshapes` / `facial_transformation_matrixes`,
    and the per-landmark `visibility`/`presence` fields are documented as
    "should stay unset if not supported" for this model, which it is not).
    So this reports 1.0 whenever a face was found at all, and the actual
    "should we trust this" decision is carried by whether a face was found
    in the first place (`None` landmarks -> `assess_risk()` already treats
    that as tracking_uncertain and forces a VLM call - see that function).
    This is reported as a real but low-information number, not a precise
    confidence score - do not read more into it than "a face was present."
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        min_face_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        output_blendshapes: bool = True,
    ):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"FaceLandmarker model not found at {model_path}. "
                f"Run download_face_landmarker_model() once (on a machine "
                f"with internet access) and copy the file to this path, or "
                f"pass model_path= explicitly."
            )

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,  # single-driver cabin camera - not a multi-face scenario
            output_face_blendshapes=output_blendshapes,
            min_face_detection_confidence=min_face_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._frame_index = 0

    def extract(self, frame_rgb: Any) -> LandmarkExtractionResult:
        """`frame_rgb` must already be RGB (e.g. the output of
        `langgraph_pipeline.enhance_frame()`, which fixes BGR->RGB as its
        first step) - feeding this a raw BGR OpenCV frame will silently
        give wrong landmark positions on an asymmetric face, the same
        color-space bug `fix_color_space()`'s docstring warns about for the
        VLM crops.
        """
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        # VIDEO mode requires a monotonically increasing timestamp per call,
        # in milliseconds - an incrementing counter is enough here since
        # this extractor owns exactly one video stream for its lifetime.
        timestamp_ms = self._frame_index * 33  # ~30fps spacing; only needs to be monotonic
        self._frame_index += 1
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return LandmarkExtractionResult(None, None, None, None)

        landmarks = result.face_landmarks[0]
        height, width = frame_rgb.shape[:2]
        points_2d = np.array([(lm.x * width, lm.y * height) for lm in landmarks], dtype=np.float64)

        left_ear = _eye_aspect_ratio(points_2d, LEFT_EYE_IDX)
        right_ear = _eye_aspect_ratio(points_2d, RIGHT_EYE_IDX)
        ear = (left_ear + right_ear) / 2.0

        yaw, pitch, roll = _head_pose(points_2d, frame_rgb.shape)

        blendshapes = None
        if getattr(result, "face_blendshapes", None):
            blendshapes = {c.category_name: c.score for c in result.face_blendshapes[0]}
        features = None
        if yaw is not None:
            features = build_features(points_2d, frame_rgb.shape, left_ear, right_ear,
                                      yaw, pitch, roll, blendshapes)

        return LandmarkExtractionResult(
            eye_aspect_ratio=ear,
            head_yaw_deg=yaw,
            head_pitch_deg=pitch,
            tracking_confidence=1.0,  # see the class docstring note above
            left_ear=left_ear,
            right_ear=right_ear,
            head_roll_deg=roll,
            features=features,
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "MeshLandmarkExtractor":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
