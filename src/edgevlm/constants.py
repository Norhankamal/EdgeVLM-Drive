"""Project-wide constants for EdgeVLM-Drive.

The two instruction strings below are copied verbatim from the experiment
notebooks. ``INSTRUCTION_V1`` was used for the Phase 1 baseline (reported in the
interim report) and for Phase 2 v1; ``INSTRUCTION_V2`` is the extended prompt used
by the final Phase 2 run. Changing a prompt changes the experiment, so keep them
immutable and add a new constant instead of editing an existing one.
"""

from __future__ import annotations

#: Class order used everywhere (confusion matrices, per-class tables, macro-F1).
LABELS: tuple[str, ...] = ("NORMAL", "DROWSY", "LOOKING_AWAY", "PHONE_USAGE")

#: Classes that should raise an alert.
UNSAFE_LABELS: frozenset[str] = frozenset({"DROWSY", "LOOKING_AWAY", "PHONE_USAGE"})

#: Label assigned when the model output cannot be mapped to one of ``LABELS``.
INVALID_LABEL = "INVALID"

#: Base model fine-tuned in this project.
MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"

#: Longest-edge pixel setting passed to the SmolVLM image processor.
IMAGE_SIZE = 512

#: Generation budget; the answer is a single label, so 12 tokens is ample.
MAX_NEW_TOKENS = 12

INSTRUCTION_V1 = (
    "You are shown two images of the same moment from an in-cabin driver camera. "
    "The first image is the full scene. The second image is a close-up crop of the "
    "driver's face, showing more detail of the eyes and head position. "
    "Using both images together, classify the driver's visible state. "
    "Return exactly one label from: NORMAL, DROWSY, LOOKING_AWAY, or PHONE_USAGE. "
    "Do not provide an explanation or punctuation."
)

INSTRUCTION_V2 = (
    "You are shown two images of the same moment from an in-cabin driver camera. "
    "The first image is the full scene, including the driver's hands, lap, and the area "
    "around the steering wheel. The second image is a close-up crop of the driver's face, "
    "showing more detail of the eyes and head position. "
    "Using both images together, classify the driver's visible state into exactly one of "
    "these four categories:\n"
    "- NORMAL: eyes on the road, hands on or near the wheel, no visible phone or device.\n"
    "- DROWSY: eyes closed, heavy eyelids, or head nodding/drooping from fatigue.\n"
    "- LOOKING_AWAY: head or eyes turned away from the road (e.g. toward a mirror, "
    "passenger, or window), with no phone or device visible in the driver's hand or lap.\n"
    "- PHONE_USAGE: a phone or handheld device is visible in the driver's hand or lap, "
    "whether the driver is looking at the road, looking down at the device, or holding it "
    "to their ear - the deciding factor is the visible device, not head pose. A driver "
    "looking down at a phone in their lap is PHONE_USAGE, not LOOKING_AWAY.\n"
    "Return exactly one label from: NORMAL, DROWSY, LOOKING_AWAY, or PHONE_USAGE. "
    "Do not provide an explanation or punctuation."
)

INSTRUCTIONS: dict[str, str] = {"v1": INSTRUCTION_V1, "v2": INSTRUCTION_V2}
