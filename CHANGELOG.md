# Changelog

## 1.0.0

- Face-mesh gate: 26-feature extractor, hard rules, any-doubt triage (`mesh_triage`) wired into the
  LangGraph pipeline.
- On-device tooling: `benchmark_jetson.py`, `evaluate_gate.py`, `simulate_stream.py`; Wilson interval and
  exact McNemar test.
- On-device results (`results/jetson`) with consistency tests.
- `unsafe_to_normal_rate` and `unsafe_class_error_rate` added to the metrics (the existing
  `unsafe_miss_rate` key is unchanged).
- Notebooks: standardized header cell, prompt-engineering notebook moved to `notebooks/prompt_engineering`.
- Documentation rewritten to match the paper; paths corrected.
