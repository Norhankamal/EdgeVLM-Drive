# SmolVLM-500M experiment log

How the four notebooks in `notebooks/finetuning_smolvlm500/` relate, what differs between the
runs, and which questions are still open. Everything here was derived from the notebooks' code and
saved outputs and from the interim report.

## Run lineage

| # | Notebook | Dataset (train / val / test) | Preprocessing | Prompt | LoRA | Validation check (macro-F1) | Sampler weight | Test macro-F1 |
|---|---|---|---|---|---|---|---|---:|
| 1 | `01_phase1_baseline_dualcrop_lora` | 17,893 / 3,941 / **3,579** | v1 | v1 | r 8 / α 16 | 160 images, patience 2 | 1/count | 74.13 % |
| 2 | `02_phase2_extended_data_v1` | 27,872 / 5,761 / **5,558** | v1 | v1 | r 16 / α 32 | 320 images, patience 2 | 1/count | 79.56 % |
| 3 | `03_phase2_extended_data_v2` | *(outputs identical to #2)* | v1 (outputs) | v1 | r 16 / α 32 | edited: 600 images, patience 3 | edited: 1/√count | *no own result* |
| 4 | `04_phase2_extended_data_final` | 27,932 / 5,761 / **5,558** | **v2** | **v2** | r 16 / α 32 | 600 images, patience 3 | 1/√count | 80.18 % |

*Preprocessing v1 / v2* = `pipeline/legacy/langgraph_pipeline_v1.py` / `pipeline/langgraph_pipeline.py`.
They differ in `extract_face_crop`: v2 additionally rejects a Haar face box whose crop has low pixel
contrast (`FACE_CROP_MIN_STD = 30`, i.e. a box sitting on a door panel or window) and falls back to a
geometric upper-region crop. Every other preprocessing function is byte-identical.
*Prompt v2* adds explicit category definitions and states that a visible phone decides
PHONE_USAGE regardless of head pose.

Other differences: runs 2-4 add weight decay 0.01; run 1 let `Trainer` also reload the best model
by `eval_loss` (this only affects the last-epoch artefact — the deployable adapter is always
`best_by_macro_f1`); run 4 adds resume-from-checkpoint and run monitoring.

Phase 1 → Phase 2 changed the **test set**: the extra data is DROWSY and NORMAL frames (test
DROWSY 168 → 1,021, test NORMAL 2,151 → 3,277; training DROWSY 1,705 → 6,287), while the
LOOKING_AWAY and PHONE_USAGE test sets are unchanged at 302 and 958. Phase 2 scores are therefore
not comparable to Phase 1.

## Observations worth knowing

- **Phase 1 per-epoch validation macro-F1:** 0.603, 0.628, **0.669**, 0.537, 0.602 — epoch 3 was kept;
  DROWSY F1 fell to 0.000 at epoch 4 before partially recovering.
- **Phase 2 v1:** best validation macro-F1 was at epoch 1 (0.794; then 0.787, 0.760), so training stopped
  after epoch 3 with no later gain.
- **Phase 2 final:** validation macro-F1 0.715, 0.757, 0.755, **0.776** over four evaluations.

## Open questions and things to verify

1. **Notebook 03 was not re-run.** Its code differs from #2 (validation subset, patience, sampler,
   texture check) but its saved outputs are identical to #2's. Do not cite it as a separate result.
2. **Final run's training length is unclear.** The notebook output says it continued from
   `checkpoint-24440` and prints `epoch 6.99…` (≈ 7 epochs of steps at 27,932 rows ÷ 8), yet the
   macro-F1 log has four epoch evaluations. The saved output cannot reconcile the two; check the
   server's `trainer_state.json` / `macro_f1_epoch_log.json` before quoting epochs or hours.
3. **Test-set information reached the Phase 2 prompt.** The comment above `INSTRUCTION` in notebook
   04 says the PHONE_USAGE/LOOKING_AWAY wording was added because the *test-set* confusion matrix
   showed PHONE_USAGE being predicted as LOOKING_AWAY. Prompt design informed by the test set makes
   the Phase 2 test numbers optimistic; a clean estimate needs a held-out set not used for any such
   decision (the validation set for design, the test set once).
4. **Confounded comparison.** Between `phase2_v1` and `phase2_final` the preprocessing, prompt,
   sampler damping, validation-subset size and patience all changed together, so the +0.6 pt
   macro-F1 — together with LOOKING_AWAY recall falling from 86.1 % to 71.2 % and the false-alarm rate
   falling from 10.22 % to 7.75 % — cannot be assigned to any one of them. Ablations would be needed.
5. **Per-class "recall" vs F1 in the interim report.** Table VI of the report is titled
   "per-class recall" and the text speaks of DROWSY *recall* collapsing to 0.0 at epoch 4. The
   callback code logs per-class **F1**, and the table values average exactly to the logged macro-F1
   (e.g. epoch 1: (0.627+0.140+0.773+0.873)/4 = 0.603). The numbers are F1; the label should be
   corrected in the report. (DROWSY F1 = 0 does imply recall = 0 at epoch 4.)
6. **Resolver and augmentation.** `FrameResolver` (ported unchanged) resolves via a split-agnostic
   suffix index as a last step, so it does not itself prevent a validation/test row from resolving
   to an augmented image. Evaluation integrity relies on the CSVs containing only original frames.
7. **Preprocessing/training alignment.** The 256M-era note in the report mentions that training and
   graph preprocessing should be aligned; for the 500M runs the dataset and evaluation both call
   the pipeline's preprocessing directly, but note that runs 1-2 used preprocessing v1 and run 4 used
   v2, so the *deployed* pipeline must use the version the chosen adapter was trained with.
8. **Latency.** GPU-server timings (≈ 315-345 ms) are single-image `generate()` on an A16. On-device
   latency is in [`results/jetson`](../../results/jetson) (≈ 6.8 s for 256M, 7.9 s for 500M per frame, 5 W mode).

## SmolVLM 256M experiments

The separate [256M experiment log](smolvlm256m_experiments.md) records that contribution without changing the 500M results above.
