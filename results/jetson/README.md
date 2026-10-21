# On-device results (NVIDIA Jetson Nano 4 GB)

| File | Contents |
|---|---|
| [`phase1_validation.json`](phase1_validation.json) | 120-image balanced diagnostic (30 per class) of the 256M and 500M DMD-data models |
| [`phase2_pilot_256m.json`](phase2_pilot_256m.json) | 80-image pilot (20 per class) of the 256M dual-crop model |
| [`phase2_balanced_1200.json`](phase2_balanced_1200.json) | main evaluation: 1,200 images (300 per class), both models, confusion matrices, latency, memory, paired comparison |
| [`gate_experiments.json`](gate_experiments.json) | face-mesh gate: hard rules and rules + triage, skipped frames per class, estimated time per frame |

## Platform and protocol

Jetson Nano 4 GB developer kit, Python 3.8, PyTorch 1.13, Transformers 4.46.0, OpenCV 4.8.0; models
in FP16 on the GPU, face mesh on the CPU. The 1,200-image runs used the 5 W power mode, which was
stable with the available power supply, so their latencies are specific to that mode. Greedy
decoding, at most 12 new tokens, and a strict parser that accepts an output only if it is exactly
one class name.

The 1,200 images were sampled at even spacing from the held-out test split of 5,558 images, 300 per
class. On a balanced set balanced accuracy equals accuracy.

## Metrics

| Metric | Definition |
|---|---|
| Unsafe-to-safe miss rate | unsafe images predicted NORMAL ÷ all unsafe images |
| Unsafe-class error rate | unsafe images not given their correct class ÷ all unsafe images |
| False-alarm rate | NORMAL images predicted as any unsafe class ÷ all NORMAL images |
| Accuracy interval | 95 % Wilson interval |
| Model comparison | exact McNemar test on paired outcomes |
| Estimated time per frame | (1 − skip rate) × model time + mesh time |

`tests/test_jetson_results.py` recomputes accuracy, macro rates, intervals, the McNemar p-value, the
throughput and the gate arithmetic from the stored confusion matrices and counts.
Reproduce a run with [`scripts/benchmark_jetson.py`](../../scripts/benchmark_jetson.py) and
[`scripts/evaluate_gate.py`](../../scripts/evaluate_gate.py).
