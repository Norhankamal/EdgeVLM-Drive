# Face-mesh gate and alert layer

A VLM call takes seconds on the Jetson Nano, so a cheap perception stage decides whether the VLM is
needed for each frame. The stage never discards a frame it cannot vouch for.

## Features (`edgevlm.mesh.features`)

MediaPipe FaceLandmarker returns 478 landmarks and blendshape scores. The extractor builds a
26-entry vector: EAR (mean, left, right, difference), yaw, pitch, roll (perspective-n-point against a
six-point face model), face centre and width, mouth aspect ratio, horizontal and vertical iris gaze
offsets, and 13 blendshape scores (eye blink, eye look in/out/up/down for each eye, jaw open, mouth
smile left/right). The mesh stage takes about 115 ms per frame on the CPU.

## Gate (`edgevlm.mesh.triage`, `pipeline.langgraph_pipeline.decision_gate_node`)

Order of checks; the first that fires calls the VLM:

| Reason | Condition |
|---|---|
| `uncertain_tracking` / `no_face` | no landmarks or no face |
| `risk_cue` | EAR < 0.20 (eyes closed) or \|yaw\| > 30° (head turned) |
| `poor_frame_quality` | blurred frame |
| `mesh_triage` | any reading outside the conservative range below, or missing |
| `scheduled_check` | `scheduled_interval_s` (default 10 s) since the last VLM call |
| *skip* | confidently normal: label NORMAL without calling the VLM |

**Conservative triage limits** (unfitted defaults chosen to be conservative, not tuned on data):
EAR ≥ 0.25, |yaw| ≤ 15°, |roll| ≤ 25°, iris gaze ≤ 0.12 horizontal / 0.15 vertical, look-down ≤ 0.30,
look-in/out/up ≤ 0.45, blink ≤ 0.45, mouth aspect ratio ≤ 0.30. The pitch limit is disabled because
camera mounting and the generic head model's offsets are unknown. Change limits with
`TriageLimits(...)`.

The scheduled check is time-based, not frame-count-based, so it keeps its meaning under adaptive
sampling. It is what covers PHONE_USAGE, which a face mesh cannot see.

## Temporal alert layer

| Class | Mechanism |
|---|---|
| DROWSY | decaying score: +1 per DROWSY frame, ×0.7 otherwise; alert at ≥ 4.0, cleared below 1.0 |
| LOOKING_AWAY | continuous elapsed time ≥ 1.5 s |
| PHONE_USAGE | one confirmed frame |

All classes share a 10 s alert cooldown. Output validation retries up to `MAX_RETRIES = 3` before
falling back to the last stable label.

## Evaluating the gate

- **Single images** (`scripts/evaluate_gate.py`): each frame takes the saved model prediction when the
  gate calls the model and NORMAL otherwise. Results: [`results/jetson/gate_experiments.json`](../results/jetson/gate_experiments.json).
- **Streams** (`scripts/simulate_stream.py`): replays a timestamped sequence through the gate, the
  scheduled check, a single-threaded VLM that is busy for one model latency per call, and the
  pipeline's own `temporal_vote_node` and `alert_decide_node`. It reports VLM calls, skipped and
  dropped frames, alerts, and the detection delay of each unsafe episode. The simulator's unit tests check, on a
  synthetic sequence, that a phone event hidden from the face mesh is detected within the scheduled
  interval plus the model time.
