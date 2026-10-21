"""
Driver Monitoring - LangGraph inference pipeline
=================================================

Implements the state graph specified in langgraph_state_design.md
(preprocess, mesh_signals, decision_gate, infer, parse, temporal_vote,
alert_decide), plus the small non-decision `mesh_result` publisher node used
on the skip branch (see the design document's states section for why that
node exists).

This module only owns the *decision logic* of the pipeline (state contract,
node functions, retry/gating rules, graph wiring). It deliberately does not
own camera I/O, MediaPipe, or the VLM runtime - those live in their own ROS
nodes / modules and are called through the small interfaces below, so this
file can be unit-tested on a dev machine without a Jetson attached.

Real integrations are marked with TODO and a short note on what needs to be
plugged in once the corresponding component is finalized.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, TypedDict


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# How many times we let `parse` send a frame back to `infer` before we give
# up and fall back to the last stable label. Keep this small - each retry
# costs a full VLM call on a 4 GB board.
MAX_RETRIES = 3

# --- Temporal-decision tuning ---
# A flat sliding-window vote treats every class the same, but DROWSY,
# LOOKING_AWAY and PHONE_USAGE have very different risk profiles and need
# different logic (see the per-class mechanism design notes). Each class
# below gets the mechanism that matches how it actually manifests on video.

# DROWSY: modeled as a decaying score rather than a fixed window, so a short
# run of DROWSY frames builds risk quickly but a single NORMAL frame doesn't
# instantly erase it (eyes can close/open within one window) while a
# sustained recovery still brings the score back down. All arithmetic below
# is O(1) per frame - trivial on a Jetson Nano 4GB.
DROWSY_SCORE_INCREMENT = 1.0       # added to the score on each DROWSY frame
DROWSY_SCORE_DECAY_FACTOR = 0.7    # score is multiplied by this on each non-DROWSY frame
DROWSY_SCORE_ALERT_THRESHOLD = 4.0   # score at/above this triggers the alert
DROWSY_SCORE_RECOVERY_THRESHOLD = 1.0  # score must drop back below this to clear the alert

# LOOKING_AWAY: eyes off the road for a short glance is normal driving
# (mirror checks, instrument cluster); what matters is elapsed time spent
# continuously looking away. 1.5s is grounded in NHTSA/Klauer-style
# eyes-off-road risk literature (see mechanisms doc references).
LOOKING_AWAY_ALERT_SECONDS = 1.5

# PHONE_USAGE: unlike the other two, a single confirmed detection is already
# unambiguous and dangerous, so it alerts immediately rather than waiting on
# an accumulated score or elapsed time.

# Shared across all three classes so the driver isn't bombarded with repeat
# alerts for a single still-ongoing event.
ALERT_COOLDOWN_S = 10.0

# How often, in seconds of wall-clock time, we force a full VLM check even if
# the tracker thinks everything is fine - this is what protects PHONE_USAGE
# detection, which facial landmarks alone cannot see. This MUST be time-based
# rather than frame-count-based (see the design document's decision-gate
# section): adaptive sampling changes how often frames arrive, so a frame
# count does not track elapsed wall-clock time. TODO: tune this once the
# adaptive sampling interval range is finalized.
DEFAULT_SCHEDULED_INTERVAL_S = 10.0

VALID_LABELS = {"NORMAL", "DROWSY", "LOOKING_AWAY", "PHONE_USAGE"}

# Mesh risk-cue thresholds, kept in one place so calibration and evaluation
# scripts read the same numbers the gate uses.
EAR_CLOSED_THRESHOLD = 0.2        # eye aspect ratio below this = eyes closed
YAW_AWAY_THRESHOLD_DEG = 30.0     # |head yaw| above this = looking away

# Any-doubt check on the mesh features. EAR/yaw alone only catch closed eyes
# and large head turns; a driver who glances away by a moderate angle or looks
# down at a phone with open eyes passes those two rules. The triage below
# sends the frame to the VLM whenever ANY mesh feature leaves a deliberately
# narrow "clearly attentive and frontal" envelope. The limits are wide-open
# defaults chosen to favour calling the VLM (they are not fitted to data);
# `None` disables a limit. Tune them with mesh_gate_study.py on recorded data.
CONSERVATIVE_TRIAGE_LIMITS = {
    "ear_min": 0.25,          # eyelids not fully open (the hard rule fires at 0.20)
    "yaw_max": 15.0,          # head turned (the hard rule fires at 30)
    "pitch_max": None,        # |pitch|; depends on camera mounting, enable after calibration
    "roll_max": 25.0,         # head tilted
    "gaze_h_max": 0.12,       # iris off-centre horizontally (fraction of eye width)
    "gaze_v_max": 0.15,       # iris off-centre vertically (looking up/down)
    "look_down_max": 0.30,    # mean eyeLookDown blendshape (reading a phone)
    "look_other_max": 0.45,   # mean of the other eyeLook* blendshapes
    "blink_max": 0.45,        # mean eyeBlink blendshape
    "mar_max": 0.30,          # mouth open (talking / yawning)
}


class ConservativeMeshTriage:
    """Model-free any-doubt check over `landmarks["mesh_features"]`.

    `assess` returns (suspicious, most_likely_unsafe_label, p_normal) like the
    learned triage in mesh_triage.py, so either can be installed with
    `set_mesh_triage`. Landmarks without a feature vector (hand-built test
    dicts) are not judged here; the extractor always provides features.
    """

    def __init__(self, limits: Optional[dict] = None):
        self.limits = dict(CONSERVATIVE_TRIAGE_LIMITS if limits is None else limits)

    def reasons(self, f: dict) -> list:
        lim, out = self.limits, []

        def over(name, value, key):
            limit = lim.get(key)
            if limit is not None and value is not None and abs(value) > limit:
                out.append(name)

        ear = f.get("ear")
        if lim.get("ear_min") is not None and ear is not None and ear < lim["ear_min"]:
            out.append("eyes_not_open")
        over("head_turned", f.get("yaw"), "yaw_max")
        over("head_pitched", f.get("pitch"), "pitch_max")
        over("head_tilted", f.get("roll"), "roll_max")
        over("gaze_sideways", f.get("gaze_h"), "gaze_h_max")
        over("gaze_vertical", f.get("gaze_v"), "gaze_v_max")
        down = [f.get("bs_eyeLookDownLeft"), f.get("bs_eyeLookDownRight")]
        if lim.get("look_down_max") is not None and all(v is not None for v in down) \
                and sum(down) / 2.0 > lim["look_down_max"]:
            out.append("looking_down")
        others = [f.get(k) for k in ("bs_eyeLookUpLeft", "bs_eyeLookUpRight", "bs_eyeLookInLeft",
                                     "bs_eyeLookInRight", "bs_eyeLookOutLeft", "bs_eyeLookOutRight")]
        if lim.get("look_other_max") is not None and all(v is not None for v in others) \
                and max(others) > lim["look_other_max"]:
            out.append("eyes_deviated")
        blink = [f.get("bs_eyeBlinkLeft"), f.get("bs_eyeBlinkRight")]
        if lim.get("blink_max") is not None and all(v is not None for v in blink) \
                and sum(blink) / 2.0 > lim["blink_max"]:
            out.append("eyelids_closing")
        over("mouth_open", f.get("mar"), "mar_max")
        return out

    def assess(self, features: Optional[dict]):
        if not features:
            return False, None, 1.0
        why = self.reasons(features)
        if not why:
            return False, None, 1.0
        if {"eyes_not_open", "eyelids_closing"} & set(why):
            guess = "DROWSY"
        elif {"head_turned", "head_tilted", "gaze_sideways", "eyes_deviated"} & set(why):
            guess = "LOOKING_AWAY"
        else:
            guess = "PHONE_USAGE"
        return True, guess, 0.0

    def suspect_table(self, df):
        """Vectorised-by-loop version for evaluation tables (same code path as `assess`)."""
        import numpy as _np
        cols = [c for c in df.columns if c in _FEATURE_COLUMNS]
        recs = df[cols].to_dict("records")
        return _np.array([bool(self.reasons({k: (None if v != v else v) for k, v in r.items()})) for r in recs])


_FEATURE_COLUMNS = frozenset(
    ["ear", "yaw", "pitch", "roll", "gaze_h", "gaze_v", "mar", "bs_eyeLookDownLeft", "bs_eyeLookDownRight",
     "bs_eyeLookUpLeft", "bs_eyeLookUpRight", "bs_eyeLookInLeft", "bs_eyeLookInRight", "bs_eyeLookOutLeft",
     "bs_eyeLookOutRight", "bs_eyeBlinkLeft", "bs_eyeBlinkRight"])

# The triage consulted by `mesh_signals_node`: ConservativeMeshTriage by default,
# a learned model from mesh_triage.py if installed, or None for the plain
# EAR/yaw rules only.
MESH_TRIAGE = ConservativeMeshTriage()


def set_mesh_triage(triage) -> None:
    """Install (or remove, with None) the triage used by `mesh_signals_node`."""
    global MESH_TRIAGE
    MESH_TRIAGE = triage


# ---------------------------------------------------------------------------
# State contract
# ---------------------------------------------------------------------------
# This mirrors the "Graph input/output schema" table in
# langgraph_state_design.md. Keeping it as a TypedDict (rather than a loose
# dict) means every node function gets type-checked field access.

class GraphState(TypedDict, total=False):
    # --- supplied by the caller (ROS vlm_node) ---
    frame_id: int
    timestamp: float
    raw_frame: Any               # numpy ndarray, kept as Any to avoid a hard numpy dependency here
    landmarks: Optional[dict]    # output of the MediaPipe landmark_node, or None if tracking is lost

    # --- filled in by preprocess ---
    full_crop: Optional[Any]
    face_crop: Optional[Any]
    frame_quality_ok: Optional[bool]  # None = not measured; False = below blur-variance threshold

    # --- filled in by mesh_signals (read by decision_gate, not recomputed there) ---
    gate_risk_cue: bool
    gate_tracking_uncertain: bool
    gate_triage_suspect: bool
    gate_triage_p_normal: Optional[float]
    mesh_provisional_label: str
    mesh_provisional_confidence: Optional[float]

    # --- filled in by decision_gate ---
    gate_decision: Literal["invoke", "skip"]
    gate_reason: str

    # --- time-based scheduled-check bookkeeping (see the design document's decision-gate section) ---
    last_vlm_timestamp: float   # updated by infer_node on each real VLM call
    scheduled_interval_s: float  # config, carried across frames by the caller

    # --- filled in by infer / parse / mesh_result ---
    vlm_raw_output: Optional[str]
    label: Optional[str]
    confidence: Optional[float]
    retry_count: int
    last_stable_label: str
    label_confirmed: bool  # False only until the first frame produces a grounded reading

    # --- carried across frames: feeds a downstream monitoring/watchdog consumer ---
    consecutive_infer_failures: int

    # --- filled in by temporal_vote / alert_decide ---
    # `last_frame_timestamp` is the wall-clock time of the previous frame,
    # used to turn LOOKING_AWAY into an elapsed-time measurement rather than
    # a frame count (frame rate varies with adaptive sampling).
    last_frame_timestamp: float
    drowsy_score: float
    drowsy_alert_active: bool
    looking_away_elapsed_s: float
    last_alert_timestamp: float
    persistent_risk: bool
    alert_triggered: bool
    alert_reason: Optional[str]  # which label triggered the alert; None if no alert

    # --- carried across every node ---
    errors: list


def new_frame_state(frame_id: int, raw_frame: Any, landmarks: Optional[dict],
                     last_stable_label: str = "NORMAL",
                     last_vlm_timestamp: float = 0.0,
                     scheduled_interval_s: float = DEFAULT_SCHEDULED_INTERVAL_S,
                     label_confirmed: bool = False,
                     consecutive_infer_failures: int = 0,
                     last_frame_timestamp: float = 0.0,
                     drowsy_score: float = 0.0,
                     drowsy_alert_active: bool = False,
                     looking_away_elapsed_s: float = 0.0,
                     last_alert_timestamp: float = 0.0,
                     timestamp: Optional[float] = None) -> GraphState:
    """Build the initial state for a single frame.

    `last_stable_label`, `last_vlm_timestamp`, `scheduled_interval_s`,
    `label_confirmed`, `consecutive_infer_failures`, `last_frame_timestamp`,
    `drowsy_score`, `drowsy_alert_active`, `looking_away_elapsed_s`, and
    `last_alert_timestamp` are all carried over from the previous frame by
    the caller (the ROS node keeps them between graph invocations) - the
    graph itself is stateless across frames on purpose, so it stays easy to
    test one frame at a time. `last_vlm_timestamp` defaults to 0.0 so the
    very first frame is always due for a scheduled check. `label_confirmed`
    defaults to False so the very first frame is honestly marked as not yet
    having a grounded reading, until either a real VLM parse or a mesh
    skip-path decision produces one (see `parse_node` / `mesh_result_node`).

    `timestamp` defaults to the current wall-clock time (live operation).
    Replay and simulation tools pass their own clock so the time-based
    scheduled check and alert cooldown behave identically offline.
    """
    return GraphState(
        frame_id=frame_id,
        timestamp=time.time() if timestamp is None else timestamp,
        raw_frame=raw_frame,
        landmarks=landmarks,
        retry_count=0,
        last_stable_label=last_stable_label,
        label_confirmed=label_confirmed,
        consecutive_infer_failures=consecutive_infer_failures,
        last_vlm_timestamp=last_vlm_timestamp,
        scheduled_interval_s=scheduled_interval_s,
        last_frame_timestamp=last_frame_timestamp,
        drowsy_score=drowsy_score,
        drowsy_alert_active=drowsy_alert_active,
        looking_away_elapsed_s=looking_away_elapsed_s,
        last_alert_timestamp=last_alert_timestamp,
        errors=[],
    )


# ---------------------------------------------------------------------------
# Small helper objects the nodes call into.
#
# These are intentionally thin interfaces, not real implementations - they
# get swapped for the actual landmark-tracking / VLM runtime integrations
# once those are wired up. Keeping them as injectable collaborators (instead
# of hardcoded calls inside the node functions) is what lets us unit-test
# the graph logic on a laptop.
# ---------------------------------------------------------------------------

@dataclass
class RiskAssessment:
    """What decision_gate reads off the MediaPipe landmarks.

    `risk_type` records WHICH cue fired (not just that one did). This matters
    even though the current gate rules make the skip path's provisional
    label unreachable when a risk cue is present (a risk cue always forces
    `invoke`, see decision_gate_node): collapsing eyes-closed and head-turned
    into a single boolean and then hard-coding the provisional label to a
    fixed value would silently mislabel a looking-away cue the moment that
    gate rule ever changes. Recording the actual risk type keeps
    `mesh_signals_node` correct-by-construction instead of correct-by-coincidence.
    """
    risk_cue: bool = False              # e.g. eyes closed too long, head turned away
    risk_type: Optional[str] = None     # "DROWSY" | "LOOKING_AWAY" | None
    tracking_uncertain: bool = False    # face partially/fully lost


def assess_risk(landmarks: Optional[dict]) -> RiskAssessment:
    """Turn raw landmark readings into a simple risk signal.

    TODO: replace this placeholder with real EAR/MAR/head-pose thresholds
    once calibrated against real footage. Until then this is a safe
    default: no landmarks -> treat tracking as uncertain (forces a VLM call
    rather than silently trusting an empty reading).
    """
    if landmarks is None:
        return RiskAssessment(risk_cue=False, risk_type=None, tracking_uncertain=True)

    ear = landmarks.get("eye_aspect_ratio")
    head_yaw = landmarks.get("head_yaw_deg")
    if ear is None or head_yaw is None:
        return RiskAssessment(risk_cue=False, risk_type=None, tracking_uncertain=True)

    eyes_closed = ear < EAR_CLOSED_THRESHOLD
    looking_away = abs(head_yaw) > YAW_AWAY_THRESHOLD_DEG

    if eyes_closed:
        return RiskAssessment(risk_cue=True, risk_type="DROWSY", tracking_uncertain=False)
    if looking_away:
        return RiskAssessment(risk_cue=True, risk_type="LOOKING_AWAY", tracking_uncertain=False)
    return RiskAssessment(risk_cue=False, risk_type=None, tracking_uncertain=False)


class VLMRuntime:
    """Thin wrapper around the on-device VLM call.

    TODO: point this at the selected compatible inference runtime, with
    grammar-constrained decoding over the four labels. For now it raises,
    so unit tests must inject a fake runtime instead of hitting real
    hardware.
    """

    def generate_label(self, face_crop: Any, full_crop: Any) -> str:
        raise NotImplementedError("Plug in the real VLM runtime here.")


class MeshInferenceAdapter:
    """The mesh-signal inference adapter, built to be swappable with
    `VLMRuntime` at the point where the graph needs "a label and a
    confidence for this frame" - not identical method signatures (a VLM
    generates free text that still needs parsing; landmark readings already
    resolve straight to a label, no parse step needed), but the same
    *contract* at the node-output boundary: both `mesh_result_node` (this
    adapter) and `parse_node` (VLMRuntime) return
    `{"label": ..., "confidence": ...}` in the same shape, so
    `temporal_vote` downstream cannot tell which adapter produced a given
    frame's reading. That shared boundary is what "swappable" means here -
    see the design document's states section for the full contract.

    Wraps `assess_risk` (the risk-classification logic) and adds the one
    piece it deliberately left out: a confidence score. `assess_risk`
    on its own only classifies; it does not say *how sure* that
    classification is, which `mesh_result_node` was previously publishing
    as a hardcoded `None` (see the TODO this replaces).
    """

    def classify(
        self, landmarks: Optional[dict], risk: Optional[RiskAssessment] = None
    ) -> tuple[str, Optional[float]]:
        """Returns (label, confidence) - the same two-value shape a parsed
        VLM output resolves to, ready to drop straight into a node's return
        dict.

        `risk` lets a caller that already computed a `RiskAssessment` (as
        `mesh_signals_node` does, since it needs the risk/tracking flags too)
        pass it straight in instead of paying for `assess_risk` twice on the
        same landmarks. Standalone callers (e.g. an adapter-contract test)
        can just pass `landmarks` and let this method compute it.

        Confidence is derived from the landmarks' own `tracking_confidence`
        field (0.0-1.0, reported by MediaPipe itself): a classification is
        only as trustworthy as the tracking it was computed from. This is a
        real, sourced number - not a placeholder - but it measures
        *tracking* quality, not classification certainty (MediaPipe Face
        Mesh has no notion of "how sure is DROWSY vs NORMAL" beyond the
        threshold crossing itself). `None` when tracking is uncertain: a
        confidence number would be meaningless attached to a reading we
        already know we shouldn't trust (see `assess_risk`'s
        `tracking_uncertain` case).
        """
        if risk is None:
            risk = assess_risk(landmarks)
        label = risk.risk_type if risk.risk_cue else "NORMAL"

        if risk.tracking_uncertain or landmarks is None:
            return label, None

        raw_confidence = landmarks.get("tracking_confidence")
        confidence = None if raw_confidence is None else max(0.0, min(1.0, float(raw_confidence)))
        return label, confidence


# Module-level default instance - the adapter has no per-call state, so one
# shared instance is enough. `mesh_signals_node` takes it as a default
# argument (not a hardcoded global reference inside the function body) so
# a test can still inject a fake/mock adapter the same way `make_infer_node`
# lets tests inject a fake `VLMRuntime` - keeping the two adapters
# swappable in tests, not just in the state-shape contract at runtime.
_default_mesh_adapter = MeshInferenceAdapter()


# ---------------------------------------------------------------------------
# Preprocessing pipeline (dual-crop) + accuracy/robustness enhancements.
#
# Every function here is plain and dependency-free (no `GraphState`, no
# LangGraph import) so the same functions used at inference time can also be
# imported directly by training-data preparation code - one shared
# implementation, not two copies kept in sync by hand.
#
# What this deliberately does NOT do: per-channel mean/std normalization.
# The selected model's own vision encoder already embeds and applies its
# expected `image_mean` / `image_std` internally when it loads an image.
# Re-normalizing here too would either be redundant or, worse, silently
# wrong if this code's constants ever drifted from the encoder's actual
# values.
#
# Pipeline order (each stage toggleable via the flags below, and each one
# benchmarked in the __main__ block so a real per-stage millisecond cost is
# visible before deciding what to keep enabled on a resource-constrained
# device):
#
#   raw_frame (BGR, OpenCV's native order)
#     -> fix_color_space          [correctness - see rationale below]
#     -> estimate_frame_quality   [read-only signal, does not modify pixels]
#     -> adaptive_gamma_correction
#     -> apply_clahe
#     -> enhanced_frame
#          -> resize_full_frame (letterbox, not crop)  -> VLM input
#          -> extract_face_crop (bbox + optional eye alignment) -> Mesh input
# ---------------------------------------------------------------------------

# Placeholder sizes pending confirmation against the selected model's actual
# expected input size once a final candidate is chosen - not values already
# confirmed against a specific model.
FULL_FRAME_SIZE = (448, 448)   # (width, height) fed to the VLM - keeps phone/body context
FACE_CROP_SIZE = (256, 256)    # (width, height) fed to the Mesh path - detail for eye/head cues

# How much margin to pad around a landmark-derived face bounding box before
# cropping, as a fraction of the box's own size - a *tight* crop risks
# cutting off the exact eye/mouth detail the crop exists to preserve.
_FACE_CROP_MARGIN = 0.35

# Minimum grayscale standard deviation a Haar-sourced face crop must have to
# be trusted as an actual face. The Haar cascade (frontal or profile) can
# report a confident-looking box over a flat, textureless surface - a door
# panel, headliner, or window - especially at the steep head angles typical
# of LOOKING_AWAY, where the cascade has the least real face texture to
# match against. A real face crop always has local contrast from eyes,
# brows, mouth, and hairline; a flat surface does not.
#
# Calibrated against the dual-crop preprocessing sanity check (fine-tuning
# notebook, Section 5): across 8 real sampled frames, every genuine
# face_crop measured std 59-76, while one Haar false positive on a
# steeply-turned LOOKING_AWAY frame (a flat door/headliner panel the
# profile cascade mistook for a face) measured std 15.7. 30.0 sits in the
# gap between those two groups with margin on both sides - not yet
# re-validated against a larger sample.
FACE_CROP_MIN_STD = 30.0

# --- Toggles: each stage's real cost is measured in the __main__ benchmark,
# not assumed - flip these off if the measured on-device latency budget
# can't absorb one of them. Nothing below is required for correctness; each
# is an accuracy lever, and levers can be turned off.
ENABLE_GAMMA_CORRECTION = True
ENABLE_CLAHE = True
ENABLE_BLUR_CHECK = True
ENABLE_FACE_ALIGNMENT = True

GAMMA_TARGET_MEAN = 128.0        # mid-gray target for adaptive gamma correction
GAMMA_MIN, GAMMA_MAX = 0.4, 2.5  # clamp so a near-black/near-white frame can't produce an extreme gamma
CLAHE_CLIP_LIMIT = 2.0           # standard CLAHE default; higher = more contrast, more noise amplification
CLAHE_TILE_GRID_SIZE = (8, 8)
BLUR_VARIANCE_THRESHOLD = 100.0  # Laplacian-variance floor below which a frame is flagged as too blurry
                                  # to trust - a literature-typical starting point (Pech-Pacheco et al.,
                                  # 2000), NOT calibrated against real driver footage yet.

# A frame whose mean brightness already falls in this range gets neither gamma
# nor CLAHE applied - both stages exist to fix poor exposure, and measurement
# shows they cost the same ~25ms whether or not the frame actually needs
# correction (a well-lit daytime frame pays the identical cost as a dark
# night frame otherwise). Skipping both on frames that don't need them cuts
# average per-frame preprocessing cost substantially on a dataset where most
# driving happens in reasonable daylight, without touching accuracy on the
# frames that do need correction.
BRIGHTNESS_OK_LOW, BRIGHTNESS_OK_HIGH = 90.0, 170.0


def fix_color_space(raw_frame: Any) -> Any:
    """Convert OpenCV's native BGR channel order to RGB.

    This is a correctness fix, not an accuracy "enhancement" - it is listed
    here anyway because it is exactly the kind of bug that silently
    tanks accuracy without ever raising an exception: every downstream
    consumer here (the VLM's own vision encoder, the landmark tracker)
    expects RGB. Camera capture through OpenCV reads frames in BGR by
    default. Feed a VLM a channel-swapped image and it does not error - it
    just quietly sees the wrong colors (skin tones shift toward blue/green),
    which can bias exactly the kind of fine visual judgment this pipeline
    depends on. This must run first, before any of the other stages below,
    all of which assume RGB.
    """
    if raw_frame is None or getattr(raw_frame, "size", 1) == 0:
        return None
    import cv2
    return cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)


def estimate_frame_quality(frame: Any) -> Optional[float]:
    """Blur score via the variance of the Laplacian - higher means sharper.

    Read-only: does not modify the frame, just measures it. A motion-blurred
    or out-of-focus frame (plausible in a moving vehicle) can fool both
    MediaPipe's landmark confidence and the VLM's own judgment without
    either one necessarily flagging low confidence on its own - this gives
    an independent, cheap (single convolution) signal the gate can use.

    Returns `None` for a missing/empty frame rather than raising.
    """
    if frame is None or getattr(frame, "size", 1) == 0:
        return None
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def adaptive_gamma_correction(frame: Any, target_mean: float = GAMMA_TARGET_MEAN) -> Any:
    """Brighten a too-dark frame or dim a too-bright one toward `target_mean`,
    via a single per-pixel lookup table (cheap: O(256) table build + one
    LUT application, not a per-pixel Python loop).

    Directly serves the goal of reliable operation across daytime and
    low-light conditions - a night-time or backlit cabin frame that reads
    very dark makes eye-closure (`DROWSY`'s key visual cue) far harder to
    distinguish from a simple shadow. Gamma is clamped to
    [GAMMA_MIN, GAMMA_MAX] so a pathological frame (e.g. a fully black
    frame from a dropped/corrupted capture) cannot produce a numerically
    unstable correction.
    """
    if frame is None or getattr(frame, "size", 1) == 0:
        return frame
    import cv2
    import numpy as np

    mean_brightness = float(frame.mean())
    if mean_brightness <= 1e-6:
        return frame  # fully black frame - nothing gamma correction can recover

    gamma = np.log(target_mean / 255.0) / np.log(mean_brightness / 255.0)
    gamma = float(np.clip(gamma, GAMMA_MIN, GAMMA_MAX))

    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype="uint8")
    return cv2.LUT(frame, lut)


def apply_clahe(frame: Any, clip_limit: float = CLAHE_CLIP_LIMIT,
                 tile_grid_size: tuple = CLAHE_TILE_GRID_SIZE) -> Any:
    """Contrast-Limited Adaptive Histogram Equalization, applied to the
    lightness channel only (LAB color space), not RGB directly.

    Plain global histogram equalization over-amplifies noise and can wash
    out a face entirely under uneven cabin lighting (e.g. sun on one side).
    CLAHE's tile-based, contrast-*limited* approach is the standard fix for
    exactly this, and is widely used in drowsiness-detection literature for
    this reason. Operating in LAB and touching only the L channel (not R/G/B
    directly) preserves color balance - color still matters here (e.g. skin
    tone vs. phone material contrast for PHONE_USAGE), so this must not be
    done as a naive per-channel equalization.
    """
    if frame is None or getattr(frame, "size", 1) == 0:
        return frame
    import cv2

    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)
    merged = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def enhance_frame(raw_frame: Any) -> tuple:
    """Runs the full correctness + accuracy-enhancement chain once, so both
    crops downstream (full-frame and face) are derived from the same
    enhanced image rather than each re-running these stages independently.

    Gamma and CLAHE are both skipped when the frame's mean brightness is
    already within [BRIGHTNESS_OK_LOW, BRIGHTNESS_OK_HIGH] - measurement
    shows both stages cost the same ~25ms whether or not the frame actually
    needed correction, so this check turns an unconditional cost into one
    paid only on the frames that benefit from it (dark/overexposed frames),
    not on every well-lit daytime frame too.

    Returns `(enhanced_frame, frame_quality_ok)` - the quality flag is
    computed on the color-corrected frame before gamma/CLAHE (a blur score
    should reflect the camera's actual optical sharpness, not stages that
    intentionally change local contrast and could distort a Laplacian-based
    measurement).
    """
    frame = fix_color_space(raw_frame)
    if frame is None:
        return None, None

    quality_score = estimate_frame_quality(frame) if ENABLE_BLUR_CHECK else None
    frame_quality_ok = None if quality_score is None else quality_score >= BLUR_VARIANCE_THRESHOLD

    needs_exposure_fix = not (BRIGHTNESS_OK_LOW <= frame.mean() <= BRIGHTNESS_OK_HIGH)

    if needs_exposure_fix:
        if ENABLE_GAMMA_CORRECTION:
            frame = adaptive_gamma_correction(frame)
        if ENABLE_CLAHE:
            frame = apply_clahe(frame)

    return frame, frame_quality_ok


def _eye_angle_deg(landmarks: dict) -> Optional[float]:
    """Angle of the line between the two eye centers, in degrees, used to
    align a tilted head so the face crop is upright.

    Expects `landmarks["left_eye_center"]` / `["right_eye_center"]` as
    (x, y) pixel coordinates. TODO: map this to the real landmark-tracker
    output once wired up. Returns `None` if either point is missing, so
    alignment degrades to a no-op rather than raising.
    """
    if not landmarks:
        return None
    left = landmarks.get("left_eye_center")
    right = landmarks.get("right_eye_center")
    if left is None or right is None:
        return None
    import math
    dx, dy = (right[0] - left[0]), (right[1] - left[1])
    return math.degrees(math.atan2(dy, dx))


def resize_full_frame(raw_frame: Any, target_size: tuple = FULL_FRAME_SIZE) -> Any:
    """Deterministic letterbox resize of the full camera frame to
    `target_size` - pads to the target aspect ratio instead of cropping.

    Changed from an earlier center-crop-to-ratio approach: cropping risks
    cutting off exactly the evidence PHONE_USAGE detection depends on when
    a hand or phone sits near the frame's edge (e.g. phone held low, near
    the lap). Letterboxing (resize to fit, pad the rest with a neutral
    gray) preserves 100% of the captured content at the cost of some
    unused border pixels - the right tradeoff when the whole point of the
    full-frame crop is scene-level context, not a tightly-framed subject.

    Returns `None` (does not raise) if `raw_frame` is `None` or empty - a
    dropped camera frame is a real, expected condition, not a bug.
    """
    if raw_frame is None or getattr(raw_frame, "size", 1) == 0:
        return None

    import cv2
    import numpy as np

    frame_h, frame_w = raw_frame.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / frame_w, target_h / frame_h)
    new_w, new_h = int(round(frame_w * scale)), int(round(frame_h * scale))

    resized = cv2.resize(raw_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_h, target_w, 3), 114, dtype=raw_frame.dtype)  # neutral gray padding
    x_off, y_off = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


# --- Lightweight fallback face locator (NOT the Mesh model) ---
#
# This is a plain OpenCV Haar-cascade face detector, bundled with opencv
# itself (no extra download, no extra dependency). It answers exactly one
# question - "is there a face-shaped region in this frame, and where" - as
# a bounding box. It does NOT produce landmarks, EAR/MAR, head-pose, or a
# tracking-confidence score, and it plays no part in `assess_risk`,
# `mesh_signals_node`, or any of the decision-gate logic. Its only job is
# to give `extract_face_crop` a real face position to crop around instead
# of a blind image-center guess, for the interim period before the actual
# Mesh model (landmark tracking) is wired in - since the face crop is fed
# to the VLM directly as part of the dual-crop input, not only to the Mesh
# path, its quality already matters before Mesh integration exists.
ENABLE_HAAR_FACE_FALLBACK = True

_haar_face_detector = None     # lazy-loaded once, not on every call
_haar_profile_detector = None  # second-pass fallback for a turned head


_SYSTEM_CASCADE_DIR = "/usr/share/opencv4/haarcascades/"


def _cascade_dir() -> str:
    import os
    import cv2
    data = getattr(cv2, "data", None)
    if data is not None:
        path = getattr(data, "haarcascades", None)
        if path and os.path.isdir(path):
            return path
    return _SYSTEM_CASCADE_DIR


def _get_haar_face_detector():
    global _haar_face_detector
    if _haar_face_detector is None:
        import cv2
        cascade_path = _cascade_dir() + "haarcascade_frontalface_default.xml"
        _haar_face_detector = cv2.CascadeClassifier(cascade_path)
    return _haar_face_detector


def _get_haar_profile_detector():
    global _haar_profile_detector
    if _haar_profile_detector is None:
        import cv2
        cascade_path = _cascade_dir() + "haarcascade_profileface.xml"
        _haar_profile_detector = cv2.CascadeClassifier(cascade_path)
    return _haar_profile_detector


def detect_face_bbox_haar(frame: Any) -> Optional[tuple]:
    """Best-effort face bounding box via Haar cascade, in `(x0, y0, x1, y1)`
    pixel coordinates - the same shape `extract_face_crop` expects from
    `landmarks["face_bbox"]`. Returns `None` if no face is found at all -
    callers must still handle that and fall back further, this is a
    best-effort locator, not a guaranteed one.

    Tries the frontal cascade first, then the profile cascade (both
    directions, via a horizontal flip) if that finds nothing. A driver
    turned toward a mirror or a rear passenger is a normal, frequent
    occurrence in this dataset (not an edge case), and the frontal-only
    cascade misses most head rotations past roughly 15-20 degrees - the
    profile pass meaningfully reduces how often callers fall through to
    `extract_face_crop`'s blind last-resort crop.

    Runs on a grayscale downscale internally (Haar cascades are grayscale
    detectors); the returned box is scaled back to the original frame's
    coordinates. When multiple faces are detected (should not happen for
    an in-cabin driver-facing camera, but a passenger could be visible),
    the largest detected box is used, since the driver is expected to be
    the dominant face in frame.
    """
    if frame is None or getattr(frame, "size", 1) == 0:
        return None

    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame

    detector = _get_haar_face_detector()
    faces = () if detector.empty() else detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        profile_detector = _get_haar_profile_detector()
        if not profile_detector.empty():
            faces = profile_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            if len(faces) == 0:
                # The profile cascade only looks one way (left-facing) -
                # flip the frame to also catch a driver turned right.
                flipped_gray = cv2.flip(gray, 1)
                flipped_faces = profile_detector.detectMultiScale(
                    flipped_gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                )
                if len(flipped_faces) > 0:
                    frame_width = gray.shape[1]
                    faces = [
                        (frame_width - x - w, y, w, h)
                        for (x, y, w, h) in flipped_faces
                    ]

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # largest by area
    return (int(x), int(y), int(x + w), int(y + h))


def extract_face_crop(
    raw_frame: Any, landmarks: Optional[dict], target_size: tuple = FACE_CROP_SIZE
) -> Any:
    """Deterministic face-region crop, guided by the landmark bounding box
    when available, resized to `target_size`.

    Bounding-box priority, best to worst:
      1. `landmarks["face_bbox"]` - the real Mesh-model box, once wired in.
      2. `detect_face_bbox_haar` - a lightweight, dependency-free fallback
         locator (see above) used whenever landmarks aren't available yet,
         so this crop is centered on an actual detected face rather than
         a blind image-center guess. This is what makes today's VLM-only
         benchmarking (no Mesh model connected) get a meaningfully
         face-centered crop instead of an arbitrary one.
      3. A centered square crop of the whole frame - the last resort when
         no face was found at all (occlusion, extreme angle, bad
         lighting), so this function still always returns *something*
         rather than raising.

    A Haar-sourced box (priority 2) is additionally checked for local
    contrast after cropping (`FACE_CROP_MIN_STD`) and demoted to priority 3
    if it looks like a flat surface rather than a face - see the check
    below for why this matters specifically for steep head angles.

    Eye alignment (`ENABLE_FACE_ALIGNMENT`) rotates the loose crop so the
    eyes are horizontal before the final resize - a head tilted even 10-15
    degrees (common while driving) otherwise skews EAR/MAR geometry, since
    those measurements assume a roughly upright face. Silently skipped
    (not an error) when eye-center landmarks aren't available - which is
    always true on the Haar/center-crop fallback paths, since neither of
    them produces eye-center points.

    Returns `None` under the same empty/missing-frame condition as
    `resize_full_frame`.
    """
    if raw_frame is None or getattr(raw_frame, "size", 1) == 0:
        return None

    import cv2
    import numpy as np

    frame_h, frame_w = raw_frame.shape[:2]
    bbox = (landmarks or {}).get("face_bbox")  # expected (x0, y0, x1, y1) in pixel coords
    bbox_from_haar = False

    if bbox is None and ENABLE_HAAR_FACE_FALLBACK:
        bbox = detect_face_bbox_haar(raw_frame)
        bbox_from_haar = bbox is not None

    def _geometric_fallback_box() -> tuple:
        # No bounding box from either landmarks or the Haar fallback (both
        # frontal and profile). Last resort: a driver's face in an in-cabin
        # dashboard frame is almost never at the frame's geometric center -
        # it sits in the upper portion, roughly centered horizontally. A
        # blind center-square crop on a wide letterboxed frame reliably
        # lands on the window or door instead of the driver.
        return (int(frame_w * 0.20), 0, int(frame_w * 0.80), int(frame_h * 0.65))

    if bbox is not None:
        x0, y0, x1, y1 = bbox
        box_w, box_h = (x1 - x0), (y1 - y0)
        mx, my = box_w * _FACE_CROP_MARGIN, box_h * _FACE_CROP_MARGIN
        x0 = max(0, int(x0 - mx))
        y0 = max(0, int(y0 - my))
        x1 = min(frame_w, int(x1 + mx))
        y1 = min(frame_h, int(y1 + my))
    else:
        x0, y0, x1, y1 = _geometric_fallback_box()

    crop = raw_frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    # The Haar cascade occasionally reports a confident-looking box over a
    # flat, textureless surface (a door panel, headliner, window) instead of
    # an actual face - a false positive, not a missing detection, and most
    # common at the steep head angles typical of LOOKING_AWAY. A real face
    # crop always has local contrast from eyes, brows, mouth, and hairline;
    # a flat surface does not. This check only applies to Haar-sourced
    # boxes - `landmarks["face_bbox"]` is the real Mesh-model box and is
    # trusted as-is once that component is wired in.
    if bbox_from_haar:
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
        if float(np.std(gray_crop)) < FACE_CROP_MIN_STD:
            x0, y0, x1, y1 = _geometric_fallback_box()
            crop = raw_frame[y0:y1, x0:x1]
            if crop.size == 0:
                return None

    if ENABLE_FACE_ALIGNMENT:
        angle = _eye_angle_deg(landmarks)
        if angle is not None:
            crop_h, crop_w = crop.shape[:2]
            center = (crop_w // 2, crop_h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            crop = cv2.warpAffine(crop, rotation_matrix, (crop_w, crop_h),
                                   borderMode=cv2.BORDER_REPLICATE)

    return cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Node functions
#
# Each node takes the current GraphState and returns a *partial* dict of the
# fields it changed - that's the LangGraph convention, and it keeps each
# node honest about what it actually owns.
# ---------------------------------------------------------------------------

def preprocess_node(state: GraphState) -> dict:
    """Turn the raw frame into the two crops the rest of the pipeline needs.

    Full-frame crop keeps phone/body context (needed for PHONE_USAGE).
    Face crop keeps the detail needed for eye-closure / head-pose cues.

    Both crops are derived from one shared `enhance_frame()` pass (color
    correction + adaptive gamma + CLAHE, each individually toggleable) so
    the two crops never disagree about brightness/contrast treatment, then
    produced by `resize_full_frame` / `extract_face_crop` - plain,
    deterministic image ops with no node-specific logic, so training-data
    preparation code can import and call the exact same functions, keeping
    the train- and inference-time preprocessing identical by construction.

    `frame_quality_ok` (from the blur-variance check) is exposed here, not
    silently dropped - `decision_gate` uses it as an additional reason to
    invoke the VLM rather than trust a skip decision built on a frame too
    blurry for MediaPipe's own reading to be reliable either.

    A `None` raw_frame (dropped camera frame - a real, expected condition,
    not a bug) degrades to `full_crop`/`face_crop` both `None` rather than
    raising; `infer_node` already treats missing tensors as an infer
    failure it can retry/fall back from, so this node does not need its own
    special-case error path.
    """
    raw_frame = state["raw_frame"]

    enhanced_frame, frame_quality_ok = enhance_frame(raw_frame)

    full_crop = resize_full_frame(enhanced_frame)
    face_crop = extract_face_crop(enhanced_frame, state.get("landmarks"))

    return {
        "full_crop": full_crop,
        "face_crop": face_crop,
        "frame_quality_ok": frame_quality_ok,
    }


def mesh_signals_node(state: GraphState, mesh_adapter: MeshInferenceAdapter = _default_mesh_adapter) -> dict:
    """Ingest the landmark-tracker output and derive everything downstream
    nodes need from it: the gate's risk/tracking signals, and the skip-path
    provisional label + confidence.

    This node does not run the tracker itself - that runs in a separate,
    external component; this just reads its published output (already
    sitting in `state["landmarks"]`) into signals the rest of the graph can
    consume, per the design document's state table.

    `mesh_adapter` defaults to the shared `MeshInferenceAdapter` instance but
    can be overridden - the same injectable-collaborator pattern
    `make_infer_node` uses for `VLMRuntime`, so tests can swap in a fake
    adapter here too (adapter contract consistency and fallback behavior
    must be testable without hardware).

    The provisional label is *computed* from the risk cue, not hard-coded to
    "NORMAL" - see the design document's decision-gate section. Under the
    current gate rules (a risk cue always forces `invoke`), a risk-cue
    frame never actually reaches the skip path, so this branch is currently
    unreachable in practice - but it stays computed, not hard-coded, so that
    invariant remains visible and auditable if the gate rules ever change.
    """
    risk = assess_risk(state.get("landmarks"))
    provisional_label, provisional_confidence = mesh_adapter.classify(state.get("landmarks"), risk=risk)

    triage_suspect, triage_p_normal = False, None
    landmarks = state.get("landmarks")
    if MESH_TRIAGE is not None and landmarks is not None and not risk.tracking_uncertain:
        # Missing features count as suspicious: a frame the triage cannot
        # read must not be skipped on its say-so.
        triage_suspect, _guess, triage_p_normal = MESH_TRIAGE.assess(landmarks.get("mesh_features"))

    return {
        "gate_risk_cue": risk.risk_cue,
        "gate_tracking_uncertain": risk.tracking_uncertain,
        "gate_triage_suspect": triage_suspect,
        "gate_triage_p_normal": triage_p_normal,
        "mesh_provisional_label": provisional_label,
        "mesh_provisional_confidence": provisional_confidence,
    }


def decision_gate_node(state: GraphState) -> dict:
    """Decide whether this frame needs a full VLM call.

    Five things force an "invoke", by design - never suppressed (the fifth
    only exists when a learned triage is installed, see `set_mesh_triage`):
      0. A learned mesh triage that does not recognise the frame as
         confidently NORMAL (`gate_triage_suspect`, reason "mesh_triage").
      1. A facial risk cue (eyes closed too long, head turned away).
      2. Uncertain or lost landmark tracking - we should not trust a
         "confidently normal" reading we don't actually have.
      3. Poor frame quality (blur-variance check, from `preprocess`) - a
         motion-blurred frame can fool the tracker's own landmark
         confidence without it necessarily reporting uncertainty, so this
         is an independent check, not redundant with #2.
      4. A periodic scheduled semantic check, because landmark tracking
         alone cannot see a phone in the driver's hand.
    Everything else is a "skip": route to `mesh_result`, which republishes
    the provisional label `mesh_signals` already computed, saving the VLM
    call.

    This node only reads `gate_risk_cue` / `gate_tracking_uncertain` - it
    does not recompute them (that is `mesh_signals`'s job, so there is one
    place, not two, that turns landmarks into a risk signal).

    The scheduled check is time-based (see the design document's
    decision-gate section), not frame-count based, so it keeps working
    under adaptive sampling: a fixed frame-count interval would drift when
    the sampling rate changes, silently breaking the PHONE_USAGE safety net
    this check exists for.
    """
    due_for_scheduled_check = (
        state["timestamp"] - state.get("last_vlm_timestamp", 0.0)
        >= state.get("scheduled_interval_s", DEFAULT_SCHEDULED_INTERVAL_S)
    )

    if state.get("gate_risk_cue"):
        return {"gate_decision": "invoke", "gate_reason": "risk_cue"}
    if state.get("gate_tracking_uncertain"):
        return {"gate_decision": "invoke", "gate_reason": "uncertain_tracking"}
    if state.get("gate_triage_suspect"):
        return {"gate_decision": "invoke", "gate_reason": "mesh_triage"}
    if state.get("frame_quality_ok") is False:
        return {"gate_decision": "invoke", "gate_reason": "poor_frame_quality"}
    if due_for_scheduled_check:
        return {"gate_decision": "invoke", "gate_reason": "scheduled_check"}

    return {"gate_decision": "skip", "gate_reason": "confidently_normal"}


# The named, traceable connection between MediaPipe-derived signals and the
# VLM invocation decision - made explicit here rather than left as an
# implicit if/elif chain only `decision_gate_node` knows about. Three of
# these four reasons originate directly from MediaPipe-derived signals
# (`mesh_signals_node`'s risk cue, tracking-uncertainty, and the
# preprocessing quality check); the fourth (`scheduled_check`) is a periodic
# safety net that fires regardless of what MediaPipe reports, specifically
# because MediaPipe landmarks cannot see a phone in the driver's hand. Any
# `gate_reason` in this set means the trigger fired and `route_after_gate`
# sends the frame to `preprocess -> infer`; anything else means it did not,
# and the frame goes to `mesh_result` instead. See the integration tests
# below, which spy on the actual VLM adapter call to prove this connection
# carries real data end to end, not just that `decision_gate_node` returns
# the right string in isolation.
VLM_TRIGGER_REASONS = frozenset({
    "risk_cue", "uncertain_tracking", "mesh_triage", "poor_frame_quality", "scheduled_check",
})


def mesh_result_node(state: GraphState) -> dict:
    """Publish the skip-path label.

    This node carries no decision logic of its own - it only copies the
    provisional label `mesh_signals` already computed into `label`, as a
    normal partial-dict node return. It exists so the skip path follows the
    same "nodes return partial dicts, routing functions don't mutate state"
    convention as every other step in this graph (routing functions should
    only choose the next node - LangGraph does not guarantee that a
    mutation made inside a routing function is merged back into the
    persisted graph state, so decision logic and state assignment must
    live in a proper node instead).
    """
    # `confidence` comes from `MeshInferenceAdapter` via `mesh_signals_node`
    # - a real value derived from the landmarks' own tracking-confidence
    # reading, not a hardcoded `None`.
    # `label_confirmed=True`: a skip-path reading is a real, grounded
    # decision from mesh_signals (no risk cue + good tracking), not a blind
    # guess - it deserves the same "trustworthy" status as a VLM-confirmed
    # label (see the design document's label-contract section).
    # `consecutive_infer_failures` is passed through unchanged rather
    # than reset to 0: a run of skipped frames says nothing about whether
    # the VLM itself is still healthy, so it should neither improve nor
    # worsen that counter.
    return {
        "label": state.get("mesh_provisional_label", "NORMAL"),
        "confidence": state.get("mesh_provisional_confidence"),
        "label_confirmed": True,
        "consecutive_infer_failures": state.get("consecutive_infer_failures", 0),
    }


def make_infer_node(vlm: VLMRuntime):
    """Factory so the node can be given a real or fake VLM runtime."""

    def infer_node(state: GraphState) -> dict:
        """Call the VLM and stash its raw output for `parse` to validate.

        Any exception here is caught rather than left to crash the graph -
        it gets recorded as a raw output of None, which `parse_node` already
        knows how to treat as a failed attempt.

        `last_vlm_timestamp` is updated here, on every entry to this node
        (including retries) - this is the actual moment a VLM call happens,
        which is what `decision_gate`'s time-based scheduled check needs to
        measure elapsed time against.

        `consecutive_infer_failures` gives a downstream monitoring/watchdog
        consumer a clean, ready-made signal instead of having to parse
        free-text `errors` strings to tell "the VLM call itself is
        crashing repeatedly" apart from "the model just returned a
        malformed label once" - a much noisier distinction to make from
        logs alone. It increments on every exception and resets to 0 the
        moment a call succeeds, regardless of whether `parse_node` later
        accepts that output as a valid label.
        """
        errors = list(state.get("errors", []))
        prior_failures = state.get("consecutive_infer_failures", 0)
        try:
            raw_output = vlm.generate_label(state.get("face_crop"), state.get("full_crop"))
            consecutive_infer_failures = 0
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            errors.append(f"infer_node failed on frame {state['frame_id']}: {exc}")
            raw_output = None
            consecutive_infer_failures = prior_failures + 1

        return {
            "vlm_raw_output": raw_output,
            "errors": errors,
            "last_vlm_timestamp": state["timestamp"],
            "consecutive_infer_failures": consecutive_infer_failures,
        }

    return infer_node


def parse_node(state: GraphState) -> dict:
    """Validate the raw VLM output against the four-class contract.

    On a bad output we do NOT immediately fall back - we ask for a retry,
    up to MAX_RETRIES, because a single malformed generation is common and
    usually recoverable. Only after the budget is exhausted do we fall back
    to the last known-stable label.
    """
    raw_output = state.get("vlm_raw_output")
    retry_count = state.get("retry_count", 0)
    errors = list(state.get("errors", []))

    label = _extract_label(raw_output)

    if label is not None:
        return {"label": label, "confidence": None, "retry_count": 0, "label_confirmed": True}

    if retry_count < MAX_RETRIES:
        errors.append(
            f"parse_node: invalid output on frame {state['frame_id']} "
            f"(attempt {retry_count + 1}/{MAX_RETRIES}): {raw_output!r}"
        )
        return {"retry_count": retry_count + 1, "errors": errors}

    # Retry budget exhausted - fall back conservatively instead of guessing.
    #
    # NOTE on the one case where this still resolves to "NORMAL" on a hard
    # failure (which the design document's failure-handling section says
    # should not generally happen): `last_stable_label` only defaults to
    # "NORMAL" via `new_frame_state`'s own default argument, before any
    # frame has ever produced a confirmed label. Because `last_vlm_timestamp`
    # also defaults to 0.0, the very first frame is always due for a
    # scheduled check (decision_gate always invokes on frame 1), so this
    # branch is reachable only if that first, mandatory VLM call ALSO
    # exhausts every retry - a narrow double-failure at system startup, not
    # the general hard-failure case.
    #
    # `label_confirmed` (see the design document's label-contract section)
    # makes that distinction a first-class, checkable value instead of
    # something only visible by reading this comment: we do NOT set it
    # here, so it stays whatever it already was. If this is genuinely the
    # first frame ever, it is still False, and the caller/watchdog can tell
    # "fallback to a real prior reading" apart from "fallback to a blind
    # default because nothing has ever been confirmed yet."
    fallback_label = state.get("last_stable_label", "NORMAL")
    startup_double_failure = not state.get("label_confirmed", False)
    errors.append(
        f"parse_node: exhausted retries on frame {state['frame_id']}, "
        f"falling back to {'unconfirmed startup default' if startup_double_failure else 'last confirmed'} "
        f"label ({fallback_label})."
    )
    return {"label": fallback_label, "confidence": None, "retry_count": 0, "errors": errors}


def _extract_label(raw_output: Optional[str]) -> Optional[str]:
    """Grammar-constrained decoding should already restrict the runtime to
    the four labels, so this is a defensive check, not the primary
    validation mechanism. Kept simple and explicit on purpose.
    """
    if not raw_output:
        return None
    candidate = raw_output.strip().upper()
    return candidate if candidate in VALID_LABELS else None


def temporal_vote_node(state: GraphState) -> dict:
    """Turn the current frame's label into a per-class risk signal.

    A flat sliding-window vote treats DROWSY, LOOKING_AWAY and PHONE_USAGE
    identically, but they manifest very differently on video, so each class
    gets the mechanism that matches it instead of one shared window:

    - DROWSY uses a decaying score rather than a fixed-size window. Eyes
      naturally blink/reopen within any short window, so a single NORMAL
      frame in the middle of a drowsy episode should not instantly reset
      the risk the way a plain vote-count would; conversely, genuine
      recovery (several NORMAL frames in a row) should still bring the
      score back down smoothly rather than requiring it to "age out" of a
      fixed window. `DROWSY_SCORE_INCREMENT` is added on each DROWSY frame;
      every other frame multiplies the running score by
      `DROWSY_SCORE_DECAY_FACTOR` instead of zeroing it outright. An alert
      state, once active, only clears when the score drops back under
      `DROWSY_SCORE_RECOVERY_THRESHOLD` - a small amount of hysteresis
      between the alert and recovery thresholds, so the alert doesn't
      flicker on and off right at the boundary.

    - LOOKING_AWAY is measured as elapsed wall-clock time spent
      continuously in that state, not a frame or vote count, because a
      normal mirror/instrument-cluster glance is a couple hundred
      milliseconds and should never alert, while anything past
      `LOOKING_AWAY_ALERT_SECONDS` of continuous eyes-off-road is the
      literature-backed risk threshold (see the mechanisms doc's NHTSA /
      Klauer et al. references). Using `last_frame_timestamp` rather than a
      count keeps this correct under adaptive frame-rate sampling, same as
      the scheduled-check interval above.

    - PHONE_USAGE has no accumulation step here at all: a single confirmed
      detection is already an unambiguous, dangerous state, so
      `alert_decide_node` fires on it immediately. There is nothing to
      compute in this node for that class.

    All of this is scalar add/multiply/compare arithmetic - O(1) per frame
    and effectively free on a Jetson Nano 4GB; the cost of this pipeline is
    entirely in the VLM call, not in this bookkeeping.
    """
    current_label = state["label"]
    now = state.get("timestamp", time.time())
    last_frame_timestamp = state.get("last_frame_timestamp", 0.0)
    dt = max(0.0, now - last_frame_timestamp) if last_frame_timestamp else 0.0

    # --- DROWSY: decaying score ---
    drowsy_score = state.get("drowsy_score", 0.0)
    if current_label == "DROWSY":
        drowsy_score += DROWSY_SCORE_INCREMENT
    else:
        drowsy_score *= DROWSY_SCORE_DECAY_FACTOR

    drowsy_alert_active = state.get("drowsy_alert_active", False)
    if drowsy_score >= DROWSY_SCORE_ALERT_THRESHOLD:
        drowsy_alert_active = True
    elif drowsy_score < DROWSY_SCORE_RECOVERY_THRESHOLD:
        drowsy_alert_active = False

    # --- LOOKING_AWAY: continuous elapsed time ---
    if current_label == "LOOKING_AWAY":
        looking_away_elapsed_s = state.get("looking_away_elapsed_s", 0.0) + dt
    else:
        looking_away_elapsed_s = 0.0

    # Kept for any downstream consumer that still wants a simple "is this
    # frame's label currently unsafe" flag (e.g. a UI indicator) - it is not
    # used to decide alerts; see alert_decide_node for the per-class logic.
    persistent_risk = (
        drowsy_alert_active
        or looking_away_elapsed_s >= LOOKING_AWAY_ALERT_SECONDS
        or current_label == "PHONE_USAGE"
    )

    return {
        "drowsy_score": drowsy_score,
        "drowsy_alert_active": drowsy_alert_active,
        "looking_away_elapsed_s": looking_away_elapsed_s,
        "last_frame_timestamp": now,
        "persistent_risk": persistent_risk,
    }


def alert_decide_node(state: GraphState) -> dict:
    """Final decision for this frame: fire the buzzer or keep monitoring.

    Each class has its own trigger condition (see `temporal_vote_node`):
    DROWSY fires once its decaying score crosses the alert threshold,
    LOOKING_AWAY fires once continuous elapsed time crosses
    `LOOKING_AWAY_ALERT_SECONDS`, and PHONE_USAGE fires immediately on a
    single confirmed frame. All three share one `ALERT_COOLDOWN_S` cooldown
    so the driver isn't bombarded with repeat alerts for what is still a
    single ongoing event.

    `last_stable_label` is only updated here, once a frame has cleared
    voting - this is the value `parse_node` falls back to on a future
    unrecoverable failure, so it should always be a label we trusted.

    Note this includes `mesh_result` (skip-path) frames, not only
    VLM-confirmed ones. A risk cue always forces `invoke`, so a skip-path
    label can only ever be `NORMAL`, never an unconfirmed unsafe guess.
    That is a statement about what the skip path can OUTPUT, not about
    whether it is always right: the mesh observes eyes and head pose, so
    DROWSY and LOOKING_AWAY surface as cues, but it cannot observe a phone.
    A PHONE_USAGE frame without a cue is skipped and reported NORMAL; the
    scheduled check bounds how long that can persist (at most one
    `scheduled_interval_s` plus one VLM call), and a phone episode shorter
    than that interval can be missed entirely. Evaluate the gate on
    streams (see mesh_gate_study.py), not on single images.
    """
    current_label = state["label"]
    now = state.get("timestamp", time.time())
    last_alert_timestamp = state.get("last_alert_timestamp", 0.0)
    cooldown_active = (now - last_alert_timestamp) < ALERT_COOLDOWN_S if last_alert_timestamp else False

    should_fire = (
        state.get("drowsy_alert_active", False)
        or state.get("looking_away_elapsed_s", 0.0) >= LOOKING_AWAY_ALERT_SECONDS
        or current_label == "PHONE_USAGE"
    )
    alert_triggered = should_fire and not cooldown_active

    return {
        "alert_triggered": alert_triggered,
        "alert_reason": current_label if alert_triggered else None,
        "last_alert_timestamp": now if alert_triggered else last_alert_timestamp,
        "last_stable_label": current_label,
    }


# ---------------------------------------------------------------------------
# Routing helpers (the conditional edges from the design doc)
# ---------------------------------------------------------------------------

def route_after_gate(state: GraphState) -> str:
    """Pure routing - no state mutation.

    Conditional-edge routing functions should only choose the next node.
    LangGraph does not guarantee that a mutation made inside a routing
    function is merged back into the persisted graph state, so decision
    logic and state assignment (choosing what `label` should be on the
    skip path) live in the `mesh_result` node instead, which returns its
    update as an ordinary partial dict like every other node in this graph.
    """
    if state["gate_decision"] == "invoke":
        return "infer"
    return "mesh_result"


def route_after_parse(state: GraphState) -> str:
    # `label` is only set once parse_node has either validated the output
    # or exhausted retries - if it's still missing, we owe infer another try.
    if state.get("label") is None:
        return "infer"
    return "temporal_vote"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

class _PlainGraph:
    def __init__(self, vlm: VLMRuntime):
        self._infer_node = make_infer_node(vlm)

    def invoke(self, state: GraphState) -> GraphState:
        state = dict(state)
        for node in (preprocess_node, mesh_signals_node, decision_gate_node):
            state.update(node(state))
        if route_after_gate(state) == "infer":
            for _ in range(MAX_RETRIES + 2):
                state.update(self._infer_node(state))
                state.update(parse_node(state))
                if route_after_parse(state) == "temporal_vote":
                    break
        else:
            state.update(mesh_result_node(state))
        for node in (temporal_vote_node, alert_decide_node):
            state.update(node(state))
        return state


def build_graph(vlm: VLMRuntime):
    return _PlainGraph(vlm)


# ---------------------------------------------------------------------------
# Minimal smoke test - run with `python langgraph_pipeline.py`.
# Uses a fake VLM so it runs on a laptop with no Jetson and no model files.
# ---------------------------------------------------------------------------

class _FakeVLM(VLMRuntime):
    """Fake VLM for the smoke test below - returns a fixed label instead of
    running a real model, so the demo works on any machine with no model
    weights required. The label is configurable per scenario (default
    "DROWSY") so a "confidently normal" walkthrough doesn't misleadingly
    report DROWSY every time the gate happens to invoke a real call.
    """

    def __init__(self, fixed_label: str = "DROWSY"):
        self.fixed_label = fixed_label

    def generate_label(self, face_crop, full_crop):
        return self.fixed_label


if __name__ == "__main__":
    print("=== Mesh inference adapter: contract consistency and fallback behavior ===")
    import numpy as np

    adapter = MeshInferenceAdapter()

    # Contract check: output shape (label, confidence) matches what parse_node
    # produces from a VLM - both are consumable identically downstream.
    label, conf = adapter.classify({"eye_aspect_ratio": 0.10, "head_yaw_deg": 0,
                                     "tracking_confidence": 0.92})
    assert label == "DROWSY" and conf == 0.92, (label, conf)
    print(f"  eyes-closed reading -> label={label!r} confidence={conf} (expected DROWSY, 0.92)")

    label, conf = adapter.classify({"eye_aspect_ratio": 0.35, "head_yaw_deg": 45,
                                     "tracking_confidence": 0.80})
    assert label == "LOOKING_AWAY" and conf == 0.80, (label, conf)
    print(f"  head-turned reading -> label={label!r} confidence={conf} (expected LOOKING_AWAY, 0.80)")

    # Fallback behavior: no landmarks at all -> uncertain tracking -> no
    # confidence number attached to a reading we already don't trust.
    label, conf = adapter.classify(None)
    assert label == "NORMAL" and conf is None, (label, conf)
    print(f"  no landmarks -> label={label!r} confidence={conf} (expected NORMAL, None)")

    # Confidence is clamped even if the landmark source ever reports outside [0, 1].
    label, conf = adapter.classify({"eye_aspect_ratio": 0.10, "head_yaw_deg": 0,
                                     "tracking_confidence": 1.4})
    assert conf == 1.0, conf
    print(f"  out-of-range tracking_confidence (1.4) clamped -> {conf} (expected 1.0)")

    print("  Mesh inference adapter: PASS\n")

    print("=== Preprocessing pipeline: determinism (train/infer consistency) ===")
    # A synthetic 480x640 BGR frame with a distinct "face" patch, so both
    # the resize and the bbox-guided crop have something real to act on -
    # not mocked, actually run through OpenCV.
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:220, 260:380] = (200, 180, 160)  # a lighter rectangular "face" region

    landmarks_with_bbox = {"face_bbox": (260, 100, 380, 220)}

    full_a = resize_full_frame(frame)
    full_b = resize_full_frame(frame)
    assert full_a.shape[:2] == (FULL_FRAME_SIZE[1], FULL_FRAME_SIZE[0])
    assert np.array_equal(full_a, full_b), "resize_full_frame is not deterministic"
    print(f"  resize_full_frame: shape={full_a.shape}, identical across two calls (deterministic)")

    face_a = extract_face_crop(frame, landmarks_with_bbox)
    face_b = extract_face_crop(frame, landmarks_with_bbox)
    assert face_a.shape[:2] == (FACE_CROP_SIZE[1], FACE_CROP_SIZE[0])
    assert np.array_equal(face_a, face_b), "extract_face_crop is not deterministic"
    print(f"  extract_face_crop (with bbox): shape={face_a.shape}, identical across two calls")

    # Fallback path: no bbox -> center crop, still deterministic, still the
    # documented output size - this is the path exercised whenever tracking
    # is uncertain (the exact condition under which we most need this to
    # not silently break).
    face_fallback = extract_face_crop(frame, None)
    assert face_fallback.shape[:2] == (FACE_CROP_SIZE[1], FACE_CROP_SIZE[0])
    print(f"  extract_face_crop (no bbox, center-crop fallback): shape={face_fallback.shape}")

    # Dropped-frame handling: None in, None out, no exception.
    assert resize_full_frame(None) is None
    assert extract_face_crop(None, landmarks_with_bbox) is None
    print("  dropped frame (raw_frame=None): both functions return None, no exception")

    print("  Preprocessing pipeline: PASS\n")

    print("=== Accuracy-enhancement preprocessing: correctness tests ===")
    import cv2

    # fix_color_space: BGR->RGB actually swaps R and B, verified on a pure-blue
    # BGR pixel (OpenCV convention: channel 0 = B) becoming a pure-red RGB pixel.
    bgr_pixel_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    bgr_pixel_frame[:, :] = (255, 0, 0)  # pure blue in BGR
    rgb_result = fix_color_space(bgr_pixel_frame)
    assert tuple(rgb_result[0, 0]) == (0, 0, 255), tuple(rgb_result[0, 0])
    print(f"  fix_color_space: BGR (255,0,0) -> RGB {tuple(rgb_result[0, 0])} (expected (0,0,255))")

    # estimate_frame_quality: a sharp high-contrast pattern must score higher
    # than a uniformly flat (blurred-equivalent) frame of the same size.
    sharp = np.zeros((200, 200, 3), dtype=np.uint8)
    sharp[::2, :] = 255  # hard alternating stripes = high-frequency content
    flat = np.full((200, 200, 3), 128, dtype=np.uint8)
    sharp_score = estimate_frame_quality(sharp)
    flat_score = estimate_frame_quality(flat)
    assert sharp_score > flat_score, (sharp_score, flat_score)
    print(f"  estimate_frame_quality: sharp={sharp_score:.1f} > flat={flat_score:.1f} (higher = sharper)")
    assert estimate_frame_quality(None) is None
    print("  estimate_frame_quality(None) -> None, no exception")

    # adaptive_gamma_correction: a too-dark frame should get brighter on average.
    dark_frame = np.full((100, 100, 3), 30, dtype=np.uint8)
    corrected = adaptive_gamma_correction(dark_frame)
    assert corrected.mean() > dark_frame.mean(), (corrected.mean(), dark_frame.mean())
    print(f"  adaptive_gamma_correction: mean brightness {dark_frame.mean():.0f} -> {corrected.mean():.0f}")

    # apply_clahe: shape/dtype preserved, output is not identical to input on
    # a frame with real local contrast variation (proves it actually did something).
    textured = np.random.default_rng(0).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    clahe_result = apply_clahe(textured)
    assert clahe_result.shape == textured.shape and clahe_result.dtype == textured.dtype
    assert not np.array_equal(clahe_result, textured)
    print(f"  apply_clahe: shape/dtype preserved ({clahe_result.shape}, {clahe_result.dtype}), pixels changed")

    # enhance_frame: full chain runs end to end and reports a quality flag.
    enhanced, quality_ok = enhance_frame(dark_frame)
    assert enhanced is not None and quality_ok is not None
    print(f"  enhance_frame: full chain OK, frame_quality_ok={quality_ok}")

    # Letterbox (not crop): a wide frame's corner content must survive resize_full_frame
    # - the old center-crop approach would have discarded it.
    wide_frame = np.zeros((200, 800, 3), dtype=np.uint8)
    wide_frame[:, 0:20] = (255, 0, 0)     # far-left marker
    wide_frame[:, -20:] = (0, 255, 0)     # far-right marker
    letterboxed = resize_full_frame(wide_frame)
    left_survived = (letterboxed[:, 0:5] == (255, 0, 0)).all(axis=-1).any()
    right_survived = (letterboxed[:, -5:] == (0, 255, 0)).all(axis=-1).any()
    assert left_survived and right_survived, "letterbox lost edge content that a crop would have discarded"
    print("  resize_full_frame: far-left/far-right markers both survived (letterbox, not crop)")

    # decision_gate: poor frame quality forces an invoke, same as risk_cue/uncertain_tracking.
    quality_state = {"gate_risk_cue": False, "gate_tracking_uncertain": False,
                      "frame_quality_ok": False, "timestamp": 0, "last_vlm_timestamp": 0,
                      "scheduled_interval_s": 999}
    gate_result = decision_gate_node(quality_state)
    assert gate_result == {"gate_decision": "invoke", "gate_reason": "poor_frame_quality"}, gate_result
    print(f"  decision_gate_node: poor_frame_quality -> {gate_result}")

    print("  Accuracy-enhancement tests: PASS\n")

    print("=== Conditional exposure fix: skip gamma/CLAHE when the frame is already well-lit ===")
    well_lit = np.random.default_rng(2).integers(100, 160, (200, 200, 3), dtype=np.uint8).astype(np.uint8)
    dark_test = np.random.default_rng(3).integers(10, 40, (200, 200, 3), dtype=np.uint8).astype(np.uint8)
    assert BRIGHTNESS_OK_LOW <= well_lit.mean() <= BRIGHTNESS_OK_HIGH, well_lit.mean()
    assert dark_test.mean() < BRIGHTNESS_OK_LOW, dark_test.mean()

    enhanced_well_lit, _ = enhance_frame(cv2.cvtColor(well_lit, cv2.COLOR_RGB2BGR))
    enhanced_dark, _ = enhance_frame(cv2.cvtColor(dark_test, cv2.COLOR_RGB2BGR))

    # Well-lit frame: gamma/CLAHE skipped -> output should equal the color-fixed
    # input (only fix_color_space ran), i.e. brightness barely moves.
    assert abs(float(enhanced_well_lit.mean()) - float(well_lit.mean())) < 2.0, (
        enhanced_well_lit.mean(), well_lit.mean())
    print(f"  well-lit frame (mean={well_lit.mean():.0f}): unchanged after enhance_frame "
          f"(mean={enhanced_well_lit.mean():.0f}) - gamma/CLAHE correctly skipped")

    # Dark frame: gamma/CLAHE still applied -> brightness should move noticeably toward target.
    assert float(enhanced_dark.mean()) > float(dark_test.mean()) + 20, (
        enhanced_dark.mean(), dark_test.mean())
    print(f"  dark frame (mean={dark_test.mean():.0f}): brightened to "
          f"mean={enhanced_dark.mean():.0f} - gamma/CLAHE correctly still applied")

    print("  Conditional exposure fix: PASS\n")

    print("=== MediaPipe-to-VLM trigger: verifying the connection reaches the VLM adapter ===")

    class SpyVLM(VLMRuntime):
        """Records every call instead of faking a label - this needs to
        prove the *connection* carries real data, not just that some fake
        label comes back. A regular `_FakeVLM` can't tell you whether it
        was even called, or with what."""
        def __init__(self):
            self.calls: list = []

        def generate_label(self, face_crop, full_crop):
            self.calls.append({"face_crop": face_crop, "full_crop": full_crop})
            return "NORMAL"

    def _quality_frame(mean_brightness: int) -> np.ndarray:
        # Enough local texture to pass the blur-variance check on its own -
        # a flat frame (uniform color) has near-zero Laplacian variance and
        # would trigger poor_frame_quality itself, masking whichever other
        # trigger a given test case is trying to isolate.
        rng = np.random.default_rng(42)
        f = rng.integers(max(0, mean_brightness - 20), min(255, mean_brightness + 20),
                          (480, 640, 3), dtype=np.int32).astype(np.uint8)
        f[100:220, 260:380] = min(255, mean_brightness + 15)
        return f

    landmarks_base = {"face_bbox": (260, 100, 380, 220), "tracking_confidence": 0.9}

    trigger_cases = [
        ("risk_cue", {**landmarks_base, "eye_aspect_ratio": 0.10, "head_yaw_deg": 0},
         _quality_frame(130), 999.0),
        ("uncertain_tracking", None, _quality_frame(130), 999.0),
        ("poor_frame_quality", {**landmarks_base, "eye_aspect_ratio": 0.35, "head_yaw_deg": 0},
         np.full((480, 640, 3), 130, dtype=np.uint8), 999.0),  # perfectly flat -> low Laplacian variance
        ("scheduled_check", {**landmarks_base, "eye_aspect_ratio": 0.35, "head_yaw_deg": 0},
         _quality_frame(130), 0.001),  # scheduled_interval_s tiny -> always due
    ]

    for expected_reason, landmarks, frame, interval_s in trigger_cases:
        spy = SpyVLM()
        pipeline = build_graph(spy)
        state = new_frame_state(frame_id=1, raw_frame=frame, landmarks=landmarks,
                                 scheduled_interval_s=interval_s)
        result = pipeline.invoke(state)

        assert result["gate_reason"] == expected_reason, (expected_reason, result["gate_reason"])
        assert expected_reason in VLM_TRIGGER_REASONS
        assert len(spy.calls) == 1, f"{expected_reason}: expected exactly 1 VLM call, got {len(spy.calls)}"
        call = spy.calls[0]
        assert call["face_crop"] is not None and call["face_crop"].shape[:2] == (
            FACE_CROP_SIZE[1], FACE_CROP_SIZE[0]), f"{expected_reason}: face_crop missing/wrong shape"
        assert call["full_crop"] is not None and call["full_crop"].shape[:2] == (
            FULL_FRAME_SIZE[1], FULL_FRAME_SIZE[0]), f"{expected_reason}: full_crop missing/wrong shape"
        print(f"  trigger={expected_reason!r}: VLM called once, with real "
              f"face_crop{call['face_crop'].shape} + full_crop{call['full_crop'].shape} - PASS")

    # Negative case: confidently normal must NOT call the VLM at all.
    spy = SpyVLM()
    pipeline = build_graph(spy)
    normal_state = new_frame_state(
        frame_id=1, raw_frame=_quality_frame(130),
        landmarks={**landmarks_base, "eye_aspect_ratio": 0.35, "head_yaw_deg": 0},
        last_vlm_timestamp=time.time(),  # pretend a VLM call just happened - otherwise every
                                          # fresh frame is "due" for its first scheduled check
                                          # by construction (last_vlm_timestamp defaults to 0.0)
        scheduled_interval_s=999.0,
    )
    result = pipeline.invoke(normal_state)
    assert result["gate_reason"] == "confidently_normal", result["gate_reason"]
    assert "confidently_normal" not in VLM_TRIGGER_REASONS
    assert len(spy.calls) == 0, f"expected 0 VLM calls on a non-trigger frame, got {len(spy.calls)}"
    assert result["label"] == result.get("mesh_provisional_label"), "skip path must publish the mesh label"
    print(f"  trigger='confidently_normal': VLM NOT called (0 calls) - PASS")

    print("  MediaPipe-to-VLM trigger checks: PASS\n")

    print("=== Real per-stage performance benchmark (this machine, not the Jetson) ===")
    import statistics

    def _bench(fn, *args, n=20):
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn(*args)
            times.append((time.perf_counter() - t0) * 1000)
        return statistics.median(times)

    bench_frame = np.random.default_rng(1).integers(0, 255, (720, 1280, 3), dtype=np.uint8).astype(np.uint8)
    bgr_bench_frame = cv2.cvtColor(bench_frame, cv2.COLOR_RGB2BGR)

    t_color = _bench(fix_color_space, bgr_bench_frame)
    rgb_bench_frame = fix_color_space(bgr_bench_frame)
    t_blur = _bench(estimate_frame_quality, rgb_bench_frame)
    t_gamma = _bench(adaptive_gamma_correction, rgb_bench_frame)
    t_clahe = _bench(apply_clahe, rgb_bench_frame)
    t_full_chain = _bench(enhance_frame, bgr_bench_frame)
    t_letterbox = _bench(resize_full_frame, rgb_bench_frame)
    t_face_crop = _bench(extract_face_crop, rgb_bench_frame, {"face_bbox": (400, 200, 800, 600)})

    well_lit_bench = np.random.default_rng(5).integers(100, 160, (720, 1280, 3), dtype=np.uint8).astype(np.uint8)
    dark_bench = np.random.default_rng(6).integers(10, 40, (720, 1280, 3), dtype=np.uint8).astype(np.uint8)
    t_chain_well_lit = _bench(enhance_frame, cv2.cvtColor(well_lit_bench, cv2.COLOR_RGB2BGR))
    t_chain_dark = _bench(enhance_frame, cv2.cvtColor(dark_bench, cv2.COLOR_RGB2BGR))

    print(f"  frame size: {bench_frame.shape[1]}x{bench_frame.shape[0]}, median of {20} runs, this dev machine:")
    print(f"    fix_color_space        {t_color:6.2f} ms")
    print(f"    estimate_frame_quality {t_blur:6.2f} ms")
    print(f"    adaptive_gamma         {t_gamma:6.2f} ms")
    print(f"    apply_clahe            {t_clahe:6.2f} ms")
    print(f"    enhance_frame (chain, random/mixed frame) {t_full_chain:6.2f} ms")
    print(f"    resize_full_frame      {t_letterbox:6.2f} ms")
    print(f"    extract_face_crop      {t_face_crop:6.2f} ms")
    print()
    print(f"    enhance_frame on a WELL-LIT frame:  {t_chain_well_lit:6.2f} ms  (gamma+CLAHE skipped)")
    print(f"    enhance_frame on a DARK frame:      {t_chain_dark:6.2f} ms  (gamma+CLAHE applied)")
    print("  NOTE: these are dev-laptop numbers for relative comparison only - the Jetson")
    print("  Nano's Cortex-A57 CPU is dramatically slower per-core. Re-run this exact")
    print("  benchmark on-device before trusting these numbers against the 0.5-2s budget,")
    print("  and disable ENABLE_CLAHE/ENABLE_GAMMA_CORRECTION/ENABLE_FACE_ALIGNMENT first")
    print("  if the on-device total doesn't fit.\n")

    pipeline = build_graph(_FakeVLM())

    print("--- Scenario 1: sustained risk cue (eyes closed) -> always invoke ---")
    last_label = "NORMAL"
    last_vlm_ts = 0.0
    label_confirmed = False
    consecutive_infer_failures = 0
    last_frame_ts = 0.0
    drowsy_score = 0.0
    drowsy_alert_active = False
    looking_away_elapsed_s = 0.0
    last_alert_ts = 0.0
    for frame_id in range(1, 6):
        state_in = new_frame_state(
            frame_id=frame_id,
            raw_frame=None,
            landmarks={"eye_aspect_ratio": 0.15, "head_yaw_deg": 5},  # simulated eyes-closed cue
            last_stable_label=last_label,
            last_vlm_timestamp=last_vlm_ts,
            scheduled_interval_s=10.0,
            label_confirmed=label_confirmed,
            consecutive_infer_failures=consecutive_infer_failures,
            last_frame_timestamp=last_frame_ts,
            drowsy_score=drowsy_score,
            drowsy_alert_active=drowsy_alert_active,
            looking_away_elapsed_s=looking_away_elapsed_s,
            last_alert_timestamp=last_alert_ts,
        )
        result = pipeline.invoke(state_in)
        last_label = result["last_stable_label"]
        last_vlm_ts = result.get("last_vlm_timestamp", last_vlm_ts)
        label_confirmed = result.get("label_confirmed", label_confirmed)
        consecutive_infer_failures = result.get("consecutive_infer_failures", consecutive_infer_failures)
        last_frame_ts = result.get("last_frame_timestamp", last_frame_ts)
        drowsy_score = result.get("drowsy_score", drowsy_score)
        drowsy_alert_active = result.get("drowsy_alert_active", drowsy_alert_active)
        looking_away_elapsed_s = result.get("looking_away_elapsed_s", looking_away_elapsed_s)
        last_alert_ts = result.get("last_alert_timestamp", last_alert_ts)
        print(
            f"frame {frame_id}: gate={result['gate_decision']} reason={result['gate_reason']} "
            f"label={result['label']} alert={result['alert_triggered']} drowsy_score={drowsy_score:.2f}"
        )

    print("\n--- Scenario 2: no risk cue, good tracking -> skip until scheduled_interval_s elapses ---")
    normal_pipeline = build_graph(_FakeVLM(fixed_label="NORMAL"))  # scheduled checks should confirm NORMAL, not DROWSY
    last_label = "NORMAL"
    last_vlm_ts = time.time()  # pretend a VLM call just happened
    label_confirmed = True  # pretend the system has already run for a while
    consecutive_infer_failures = 0
    last_frame_ts = 0.0
    drowsy_score = 0.0
    drowsy_alert_active = False
    looking_away_elapsed_s = 0.0
    last_alert_ts = 0.0
    for frame_id in range(1, 4):
        state_in = new_frame_state(
            frame_id=frame_id,
            raw_frame=None,
            landmarks={"eye_aspect_ratio": 0.35, "head_yaw_deg": 2},  # confidently normal
            last_stable_label=last_label,
            last_vlm_timestamp=last_vlm_ts,
            scheduled_interval_s=1.0,  # short interval so the demo shows both branches quickly
            label_confirmed=label_confirmed,
            consecutive_infer_failures=consecutive_infer_failures,
            last_frame_timestamp=last_frame_ts,
            drowsy_score=drowsy_score,
            drowsy_alert_active=drowsy_alert_active,
            looking_away_elapsed_s=looking_away_elapsed_s,
            last_alert_timestamp=last_alert_ts,
        )
        result = normal_pipeline.invoke(state_in)
        last_label = result["last_stable_label"]
        last_vlm_ts = result.get("last_vlm_timestamp", last_vlm_ts)
        label_confirmed = result.get("label_confirmed", label_confirmed)
        consecutive_infer_failures = result.get("consecutive_infer_failures", consecutive_infer_failures)
        last_frame_ts = result.get("last_frame_timestamp", last_frame_ts)
        drowsy_score = result.get("drowsy_score", drowsy_score)
        drowsy_alert_active = result.get("drowsy_alert_active", drowsy_alert_active)
        looking_away_elapsed_s = result.get("looking_away_elapsed_s", looking_away_elapsed_s)
        last_alert_ts = result.get("last_alert_timestamp", last_alert_ts)
        print(
            f"frame {frame_id}: gate={result['gate_decision']} reason={result['gate_reason']} "
            f"label={result['label']} alert={result['alert_triggered']}"
        )
        time.sleep(1.1)  # exceed scheduled_interval_s so the next frame is due for a check

    print("\n--- Scenario 3: VLM adapter keeps raising -> consecutive_infer_failures climbs, "
          "label_confirmed stays False on a cold start ---")

    class _AlwaysCrashesVLM(VLMRuntime):
        def generate_label(self, face_crop, full_crop):
            raise RuntimeError("simulated runtime crash (e.g. OOM on the 4 GB Nano)")

    crashing_pipeline = build_graph(_AlwaysCrashesVLM())
    state_in = new_frame_state(
        frame_id=1,
        raw_frame=None,
        landmarks={"eye_aspect_ratio": 0.15, "head_yaw_deg": 5},  # force invoke via risk cue
        scheduled_interval_s=10.0,
        # label_confirmed defaults to False: this is a genuine cold start, no prior frame ever ran.
    )
    result = crashing_pipeline.invoke(state_in)
    print(
        f"frame 1 (cold start, VLM always crashes): label={result['label']} "
        f"label_confirmed={result['label_confirmed']} "
        f"consecutive_infer_failures={result['consecutive_infer_failures']}"
    )
    print("(label_confirmed=False flags this NORMAL as an unconfirmed startup default, "
          "not a grounded reading - see the design document's label-contract section)")
