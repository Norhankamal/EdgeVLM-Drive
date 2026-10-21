from __future__ import annotations

from dataclasses import dataclass

from dms_ros import dms_preprocess as pipeline

NAN = float("nan")


@dataclass
class GateDecision:
    decision: str
    reason: str
    face_found: bool
    ear: float
    yaw: float
    pitch: float
    roll: float


class MeshGate:
    """The decision gate of the original mesh pipeline, with its own constants:
    risk cue (EAR / yaw), any-doubt triage, frame quality, scheduled check."""

    def __init__(self):
        self.ear_closed = pipeline.EAR_CLOSED_THRESHOLD
        self.yaw_away = pipeline.YAW_AWAY_THRESHOLD_DEG
        self.triage = pipeline.ConservativeMeshTriage()

    def evaluate(self, landmarks, quality_ok, now, last_invoke_t, scheduled_interval_s):
        if landmarks is None:
            return self._decision("invoke", "uncertain_tracking", False, NAN, NAN, NAN, NAN)

        ear = landmarks.get("eye_aspect_ratio")
        yaw = landmarks.get("head_yaw_deg")
        if ear is None or yaw is None:
            return self._decision("invoke", "uncertain_tracking", False, NAN, NAN, NAN, NAN)

        features = landmarks.get("mesh_features") or {}
        pitch = features.get("pitch", NAN)
        roll = features.get("roll", NAN)

        if ear < self.ear_closed or abs(yaw) > self.yaw_away:
            return self._decision("invoke", "risk_cue", True, ear, yaw, pitch, roll)

        suspect, _guess, _p_normal = self.triage.assess(features)
        if suspect:
            return self._decision("invoke", "mesh_triage", True, ear, yaw, pitch, roll)

        if quality_ok is False:
            return self._decision("invoke", "poor_frame_quality", True, ear, yaw, pitch, roll)

        if scheduled_interval_s > 0.0 and now - last_invoke_t >= scheduled_interval_s:
            return self._decision("invoke", "scheduled_check", True, ear, yaw, pitch, roll)

        return self._decision("skip", "confidently_normal", True, ear, yaw, pitch, roll)

    @staticmethod
    def _decision(decision, reason, face_found, ear, yaw, pitch, roll):
        return GateDecision(decision, reason, face_found, float(ear), float(yaw), float(pitch), float(roll))
