# SmolVLM 256M experiment progression

## Initial adaptation and prompt formatting

We began with single-image SmolVLM-256M and language-decoder attention LoRA. The initial notebook includes a short smoke test and a separate full training cell. Prompt wording and strict text-label formatting were investigated; valid output syntax does not establish correct recognition. The prompt-benchmark source is preserved in the notebook archive, without inventing a new result for cells without matching executed evidence.

## Augmentation and imbalance treatments

Training used offline augmented examples and duplicate-path checks. Inverse-frequency sampling and class-weighted loss were alternative experiments, not automatically combined. The early LoRA configuration uses rank 8, alpha 16 and dropout 0.05 on text attention. Historical direct results in notebooks are separate from the downloaded direct prediction exports. The graph evaluations create a fresh state for each image; their results do not demonstrate temporal alert behavior or reduced VLM calls. DROWSY recall remained very low in the exported single-image graph results.

## Paired scene and face inputs

Intermediate extended-data and ETA notebooks are retained before the dual-crop baseline. The baseline pairs a full-scene view with a face crop and adapts attention in both vision and language branches with rank 16, alpha 32 and dropout 0.05. It uses the earlier merged-adapter initialization, inverse-frequency sampling, a 1e-4 learning rate and generated-label validation macro F1 for model selection (80 examples per class, patience 2, maximum 8 epochs). The later test exports have 5,558 rows instead of 3,579. Dataset preparation is outside this contribution.

## Revised training without MediaPipe

The revised model starts from the pretrained base, uses a 5e-5 learning rate, square-root inverse-frequency sampling, and mild training-only brightness/contrast augmentation for LOOKING_AWAY. Prompt definitions distinguish visible phone evidence from looking away. Validation uses up to 150 examples per class, patience 2 and a maximum of 6 epochs. Its training copy has no saved executed cells; the executed evaluation notebook and prediction CSV establish the scoring result.

## MediaPipe crop and facial cue experiment

MediaPipe Face Landmarker guides the face box, with fallback behavior in the notebook, and cached blendshape scores provide blink, gaze and jaw-opening text cues. This is crop/cue preparation for both training and inference, not an inference gate. This configuration also restores earlier-adapter initialization, uses learning rate 1e-4, augments DROWSY and LOOKING_AWAY, and uses validation up to 150 per class with patience 3 and an 8-epoch ceiling. Checkpoints are requested every 100 steps. These simultaneous changes prevent assigning the score difference to MediaPipe alone.

The common later training machinery includes FP16, gradient checkpointing, batch size 1 with 8 accumulation steps, AdamW, cosine scheduling, 5% warm-up, weight decay 0.01, seed 42 and answer-token-only loss. Greedy generation is parsed into the four labels or INVALID. Validation selects an adapter; the final adapter and resume checkpoints are not interchangeable. Source configuration records preserve the exact expressions rather than merging variants.

## Results and interpretation

| Experiment | Images | Accuracy % | Balanced accuracy % | Macro F1 % | DROWSY recall % |
|---|---:|---:|---:|---:|---:|
| `single_image_sampling_direct` | 3,579 | 78.09 | 51.31 | 51.92 | 0.00 |
| `single_image_weighted_direct` | 3,579 | 71.86 | 58.59 | 52.90 | 5.95 |
| `single_image_sampling_graph` | 3,579 | 79.46 | 48.81 | 50.78 | 0.00 |
| `single_image_weighted_graph` | 3,579 | 78.90 | 58.90 | 55.51 | 2.38 |
| `dual_crop_baseline` | 5,558 | 83.68 | 82.80 | 78.16 | 87.07 |
| `revised_no_mediapipe` | 5,558 | 85.16 | 81.89 | 79.63 | 86.88 |
| `mediapipe` | 5,558 | 85.98 | 82.46 | 80.61 | 85.01 |

Full-cohort scores describe each exported test population. The matched-path comparison below removes the extra test rows but does not align training settings or establish historical image-content identity.

| Experiment | Images | Accuracy % | Balanced accuracy % | Macro F1 % | DROWSY recall % |
|---|---:|---:|---:|---:|---:|
| `single_image_sampling_direct` | 3,579 | 78.09 | 51.31 | 51.92 | 0.00 |
| `single_image_weighted_direct` | 3,579 | 71.86 | 58.59 | 52.90 | 5.95 |
| `single_image_sampling_graph` | 3,579 | 79.46 | 48.81 | 50.78 | 0.00 |
| `single_image_weighted_graph` | 3,579 | 78.90 | 58.90 | 55.51 | 2.38 |
| `dual_crop_baseline` | 3,579 | 84.07 | 78.01 | 73.08 | 63.10 |
| `revised_no_mediapipe` | 3,579 | 84.94 | 75.13 | 73.43 | 55.36 |
| `mediapipe` | 3,579 | 86.31 | 75.08 | 74.87 | 50.60 |

The expanded cohort adds 1,126 NORMAL and 853 DROWSY paths. Drowsiness recall on the shared population remains lower than on the full later population, so the larger-cohort score alone overstates recognition on the earlier drowsy examples. Aggregate improvement should be discussed together with class-specific misses and false alarms.

## Reproducibility limits

- Prediction hashes identify the scored exports, not the model weights that generated every row. Cached prediction reuse may obscure that provenance.
- No train/validation/test split files or driver images are included or independently audited in this update.
- Historical notebook scores and later CSV scores remain distinct records. Draft and partially executed notebooks are not promoted to completed experiments.
- MediaPipe validation history contains repeated near-epoch-5 evaluations after resume. They are retained as recorded, not counted as unique epochs.
- Helper snapshots are the downloaded versions; exact preprocessing at each historical run still requires a contemporaneous hash.
- Single runs with changing settings do not establish causal effects or confidence intervals. Recorded generation and graph latency are different measurements.
- No new GPU training, model inference, merged export, quantization or live-system validation was performed for this repository update.

For numerical definitions and recalculation commands, see [the results guide](../results/smolvlm_256m/README.md). For original-file mapping and execution status, see [the notebook guide](../notebooks/finetuning_smolvlm256/README.md).

## Validation of this repository addition

The new saved-prediction scorer is covered by 15 CPU-only regression tests, including malformed inputs, duplicate paths, label mismatches, INVALID predictions, metric reconciliation and CLI overwrite protection. Notebook execution counts remain those of the downloaded files. The training-metadata archive containing trainer_state.json was not present among the supplied downloads; checkpoint-state claims are therefore not added. No source notebook was executed as part of packaging.

The targeted validation run passed 17 tests: the 15 new scorer tests plus the existing prompt/notebook checks, including notebook-schema validation. Lint passed for the new Python files. The full local suite had 34 passes and 13 failures: three existing Linux-style absolute-path assertions fail on Windows, and ten existing tests require scikit-learn, which is absent from this validation environment. Those unrelated implementation files were not changed. The existing notebook test directory spelling was corrected to match the cloned repository. Full GPU execution remains untested.
