import numpy as np

from edgevlm.edge.stream_sim import StreamFrame, simulate_stream
from edgevlm.mesh.features import FEATURE_NAMES


def calm():
    f = {n: 0.0 for n in FEATURE_NAMES}
    f.update(ear_mean=0.32, yaw=1.0, roll=0.0, mar=0.05, gaze_h=0.01, gaze_v=0.01)
    return f


def sequence(seconds, phone_from=None, step=1.0):
    frames = []
    for t in np.arange(0, seconds, step):
        phone = phone_from is not None and t >= phone_from
        frames.append(StreamFrame(t=float(t), true_label="PHONE_USAGE" if phone else "NORMAL",
                                  features=calm(), vlm_label="PHONE_USAGE" if phone else "NORMAL"))
    return frames


def test_normal_driving_skips_most_frames_and_raises_no_alert():
    r = simulate_stream(sequence(60), vlm_latency_s=2.0, scheduled_interval_s=10.0)
    assert r.alerts == [] and r.skipped > 0
    # a scheduled call is forced about every 10 s, so far fewer VLM calls than frames
    assert 4 <= r.vlm_calls <= 8 and r.vlm_calls < r.processed / 3
    assert r.gate_reasons["scheduled_check"] == r.vlm_calls - r.gate_reasons.get("mesh_triage", 0)


def test_phone_use_that_the_face_cannot_show_is_caught_by_the_scheduled_check_within_a_bound():
    interval, latency = 10.0, 3.0
    r = simulate_stream(sequence(90, phone_from=23.0), vlm_latency_s=latency, scheduled_interval_s=interval)
    ep = next(e for e in r.episodes if e["label"] == "PHONE_USAGE")
    assert ep["detected"]
    assert ep["detection_delay_s"] <= interval + latency + 1.0   # bound: schedule interval + model time + sampling step


def test_busy_vlm_drops_frames_that_arrive_during_a_call():
    frames = sequence(20)
    for f in frames:
        f.features = {**calm(), "yaw": 20.0}        # triage doubt on every frame -> VLM every time
    r = simulate_stream(frames, vlm_latency_s=4.5, scheduled_interval_s=10.0)
    assert r.dropped_busy > 0 and r.processed + r.dropped_busy == r.frames
    assert r.vlm_calls == r.processed


def test_drowsy_alert_needs_sustained_detections():
    frames = [StreamFrame(t=float(t), true_label="DROWSY", features={**calm(), "ear_mean": 0.1}, vlm_label="DROWSY")
              for t in range(0, 40, 1)]
    r = simulate_stream(frames, vlm_latency_s=1.0)
    assert any(a.reason == "DROWSY" for a in r.alerts)
    first = min(a.t for a in r.alerts if a.reason == "DROWSY")
    assert first > 3.0   # the decaying score needs several DROWSY frames, a single frame must not alert
