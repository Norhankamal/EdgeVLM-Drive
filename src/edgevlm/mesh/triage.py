"""Model-free gate rules (paper, Section IV-C).

``hard_rule_reasons`` are the original rules: no face, eyes closed (EAR < 0.20), head turned
(|yaw| > 30 degrees) or a blurred frame each force a VLM call.

``ConservativeMeshTriage`` adds the *any-doubt* check: a frame counts as confidently NORMAL only if
every conservative condition holds; a missing reading or any single doubtful reading sends the frame
to the VLM (reason ``mesh_triage``). The limits are unfitted, conservative defaults, not values tuned
on data. The pitch limit is disabled because camera mounting and the generic head model's offsets are
unknown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

_LOOK_OTHER = ("eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
               "eyeLookUpLeft", "eyeLookUpRight")
_LOOK_DOWN = ("eyeLookDownLeft", "eyeLookDownRight")
_BLINK = ("eyeBlinkLeft", "eyeBlinkRight")


@dataclass(frozen=True)
class HardRuleLimits:
    ear_closed: float = 0.20
    yaw_max_deg: float = 30.0


@dataclass(frozen=True)
class TriageLimits:
    ear_min: float = 0.25
    yaw_max_deg: float = 15.0
    roll_max_deg: float = 25.0
    gaze_h_max: float = 0.12
    gaze_v_max: float = 0.15
    look_down_max: float = 0.30
    look_other_max: float = 0.45
    blink_max: float = 0.45
    mar_max: float = 0.30
    pitch_max_deg: Optional[float] = None  # disabled by default (see module docstring)


@dataclass(frozen=True)
class GateDecision:
    invoke: bool
    # no_face | eyes_closed | head_turned | poor_frame_quality | mesh_triage | scheduled_check | confidently_normal
    reason: str


def hard_rule_reasons(
    features: Optional[Mapping[str, float]],
    blurred: bool = False,
    limits: HardRuleLimits = HardRuleLimits(),
) -> list[str]:
    """Reasons from the original hard rules that force a VLM call (empty list = none fired)."""
    reasons: list[str] = []
    if features is None:
        return ["no_face"]
    if features["ear_mean"] < limits.ear_closed:
        reasons.append("eyes_closed")
    if abs(features["yaw"]) > limits.yaw_max_deg:
        reasons.append("head_turned")
    if blurred:
        reasons.append("poor_frame_quality")
    return reasons


@dataclass
class ConservativeMeshTriage:
    """Any-doubt triage over the 26-feature vector."""

    limits: TriageLimits = field(default_factory=TriageLimits)

    def doubts(self, features: Optional[Mapping[str, float]]) -> list[str]:
        """Names of the conditions that are NOT satisfied (empty list = confidently normal)."""
        if features is None:
            return ["no_features"]
        lim, f = self.limits, features
        doubts: list[str] = []

        def get(name: str) -> Optional[float]:
            v = f.get(name)
            return None if v is None else float(v)

        checks = [
            ("ear", get("ear_mean"), lambda v: v >= lim.ear_min),
            ("yaw", get("yaw"), lambda v: abs(v) <= lim.yaw_max_deg),
            ("roll", get("roll"), lambda v: abs(v) <= lim.roll_max_deg),
            ("gaze_h", get("gaze_h"), lambda v: abs(v) <= lim.gaze_h_max),
            ("gaze_v", get("gaze_v"), lambda v: abs(v) <= lim.gaze_v_max),
            ("mar", get("mar"), lambda v: v <= lim.mar_max),
        ]
        if lim.pitch_max_deg is not None:
            checks.append(("pitch", get("pitch"), lambda v: abs(v) <= lim.pitch_max_deg))
        for name, value, ok in checks:
            if value is None or not ok(value):
                doubts.append(name)

        for name, keys, limit in (("look_down", _LOOK_DOWN, lim.look_down_max),
                                  ("look_other", _LOOK_OTHER, lim.look_other_max),
                                  ("blink", _BLINK, lim.blink_max)):
            values = [get(k) for k in keys]
            if any(v is None for v in values) or max(values) > limit:
                doubts.append(name)
        return doubts

    def confident_normal(self, features: Optional[Mapping[str, float]]) -> bool:
        return not self.doubts(features)


def decide_gate(
    features: Optional[Mapping[str, float]],
    blurred: bool = False,
    seconds_since_vlm: float = 0.0,
    scheduled_interval_s: float = 10.0,
    use_triage: bool = True,
    triage: Optional[ConservativeMeshTriage] = None,
    hard: HardRuleLimits = HardRuleLimits(),
) -> GateDecision:
    """Decide whether a frame needs the VLM. Order: hard rules, triage, scheduled check, otherwise skip."""
    reasons = hard_rule_reasons(features, blurred=blurred, limits=hard)
    if reasons:
        return GateDecision(True, reasons[0])
    if use_triage and not (triage or ConservativeMeshTriage()).confident_normal(features):
        return GateDecision(True, "mesh_triage")
    if seconds_since_vlm >= scheduled_interval_s:
        return GateDecision(True, "scheduled_check")
    return GateDecision(False, "confidently_normal")
