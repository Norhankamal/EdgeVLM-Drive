"""Stream simulator: runs the gate, the scheduled check and the temporal alert layer on a recorded sequence.

A single-image evaluation cannot include the scheduled VLM check or any temporal behaviour (paper,
Section II-F). This simulator replays a timestamped sequence through

1. the decision gate (hard rules, any-doubt triage, scheduled check),
2. a single-threaded VLM worker that is busy for ``vlm_latency_s`` per call (frames that arrive while
   it is busy are not processed), and
3. the pipeline's own ``temporal_vote_node`` and ``alert_decide_node``,

and reports how many VLM calls were made, which alerts fired and how long each unsafe episode took
to be detected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

from edgevlm.constants import UNSAFE_LABELS
from edgevlm.mesh.triage import ConservativeMeshTriage, decide_gate

_T0 = 1000.0  # keeps simulated timestamps away from the "0.0 means unset" convention of the pipeline nodes


@dataclass
class StreamFrame:
    t: float                                   # seconds from the start of the sequence
    true_label: str
    features: Optional[Mapping[str, float]]    # 26-feature dict, or None when no face was found
    vlm_label: str                             # what the VLM would answer for this frame
    blurred: bool = False


@dataclass
class Alert:
    t: float
    reason: str


@dataclass
class StreamResult:
    frames: int
    processed: int
    dropped_busy: int
    vlm_calls: int
    skipped: int
    gate_reasons: dict
    alerts: list[Alert] = field(default_factory=list)
    episodes: list[dict] = field(default_factory=list)

    @property
    def skip_rate(self) -> float:
        return self.skipped / self.processed if self.processed else 0.0


def _episodes(frames: Sequence[StreamFrame]) -> list[dict]:
    """Maximal runs of the same unsafe true label."""
    out, start, label = [], None, None
    for fr in frames:
        if fr.true_label in UNSAFE_LABELS and fr.true_label == label:
            continue
        if label is not None:
            out.append({"label": label, "start": start, "end": fr.t})
            label = None
        if fr.true_label in UNSAFE_LABELS:
            label, start = fr.true_label, fr.t
    if label is not None:
        out.append({"label": label, "start": start, "end": frames[-1].t})
    return out


def simulate_stream(
    frames: Sequence[StreamFrame],
    vlm_latency_s: float,
    scheduled_interval_s: float = 10.0,
    use_triage: bool = True,
    triage: Optional[ConservativeMeshTriage] = None,
    mesh_latency_s: float = 0.0,
    on_alert: Optional[Callable[[Alert], None]] = None,
) -> StreamResult:
    """Replay ``frames`` (sorted by time) and return counts, alerts and per-episode detection delays."""
    from edgevlm.pipeline.langgraph_pipeline import alert_decide_node, temporal_vote_node

    frames = sorted(frames, key=lambda f: f.t)
    state: dict = {"last_frame_timestamp": 0.0, "drowsy_score": 0.0, "drowsy_alert_active": False,
                   "looking_away_elapsed_s": 0.0, "last_alert_timestamp": 0.0}
    busy_until = -1.0
    last_vlm_t = None
    reasons: dict = {}
    result = StreamResult(frames=len(frames), processed=0, dropped_busy=0, vlm_calls=0, skipped=0, gate_reasons=reasons)

    for fr in frames:
        if fr.t < busy_until:
            result.dropped_busy += 1
            continue
        result.processed += 1
        since = float("inf") if last_vlm_t is None else fr.t - last_vlm_t
        decision = decide_gate(fr.features, blurred=fr.blurred, seconds_since_vlm=since,
                               scheduled_interval_s=scheduled_interval_s, use_triage=use_triage, triage=triage)
        reasons[decision.reason] = reasons.get(decision.reason, 0) + 1

        if decision.invoke:
            result.vlm_calls += 1
            label, decided_at = fr.vlm_label, fr.t + mesh_latency_s + vlm_latency_s
            busy_until, last_vlm_t = decided_at, fr.t
        else:
            result.skipped += 1
            label, decided_at = "NORMAL", fr.t + mesh_latency_s

        state.update(label=label, timestamp=_T0 + decided_at)
        state.update(temporal_vote_node(state))
        update = alert_decide_node(state)
        state.update(update)
        if update["alert_triggered"]:
            alert = Alert(t=decided_at, reason=update["alert_reason"])
            result.alerts.append(alert)
            if on_alert:
                on_alert(alert)

    for ep in _episodes(frames):
        window_end = ep["end"] + vlm_latency_s
        hits = [a.t for a in result.alerts if a.reason == ep["label"] and ep["start"] <= a.t <= window_end]
        ep["detected"] = bool(hits)
        ep["detection_delay_s"] = (hits[0] - ep["start"]) if hits else None
        result.episodes.append(ep)
    return result
