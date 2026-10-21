# SmolVLM-500M results

| File | Contents |
|---|---|
| `phase1_baseline.json` | Phase 1 test metrics, per-class table, confusion matrix, per-epoch validation history (the run reported in the interim report) |
| `phase2_v1.json` | Phase 2 v1 (extended data, LoRA r = 16) |
| `phase2_final.json` | Phase 2 final (current preprocessing and prompt) |
| [`../jetson/`](../jetson) | on-device results (Jetson Nano) for the 256M and 500M models |

The three `phase*.json` files were extracted programmatically from the saved outputs of the
evaluation cell of the corresponding notebook (`source.notebook` in each file). Percentages are
as printed by the notebooks (2 decimals); per-class values have 3 decimals. `tests/test_results_consistency.py`
recomputes every headline metric from the stored confusion matrix and checks it matches.

Per-class values in `validation_epoch_history` are per-class **F1** (as logged by the callback).
Phase 2 test sets differ from Phase 1's, so compare runs only within a phase — see
[`docs/experiment_log.md`](../../docs/experiment_log.md).
