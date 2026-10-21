"""Face-mesh perception: the 26-feature vector and the model-free gate rules."""

from edgevlm.mesh.features import FEATURE_NAMES, MeshFeatureExtractor, eye_aspect_ratio, mouth_aspect_ratio
from edgevlm.mesh.triage import (
    ConservativeMeshTriage,
    GateDecision,
    HardRuleLimits,
    TriageLimits,
    decide_gate,
    hard_rule_reasons,
)

__all__ = [
    "FEATURE_NAMES",
    "MeshFeatureExtractor",
    "eye_aspect_ratio",
    "mouth_aspect_ratio",
    "ConservativeMeshTriage",
    "GateDecision",
    "HardRuleLimits",
    "TriageLimits",
    "decide_gate",
    "hard_rule_reasons",
]
