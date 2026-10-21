# SmolVLM 256M notebooks

Read notebooks 01 through 12 as the development sequence. The archive directory contains source variants and the prompt-benchmark notebook, not additional scored runs. No cells were executed during repository preparation.

| Repository notebook | Original download | Saved executed code cells |
|---|---|---:|
| [01_initial_single_image.ipynb](01_initial_single_image.ipynb) | `EdgeVLM_finetuning_fixed.ipynb` | 20/22 |
| [02_augmented_sampling.ipynb](02_augmented_sampling.ipynb) | `EdgeVLM_finetuning_augmented_complete (1).ipynb` | 19/21 |
| [03_augmented_classweighted.ipynb](03_augmented_classweighted.ipynb) | `EdgeVLM_finetuning_augmented_complete_classweighted.ipynb` | 21/26 |
| [04_sampling_langgraph.ipynb](04_sampling_langgraph.ipynb) | `EdgeVLM_finetuning_vlm256_langgraph (1).ipynb` | 23/27 |
| [05_classweighted_langgraph.ipynb](05_classweighted_langgraph.ipynb) | `EdgeVLM_finetuning_vlm256_classweighted (2).ipynb` | 24/30 |
| [06_extended_data_single_gpu.ipynb](06_extended_data_single_gpu.ipynb) | `EdgeVLM_256M_Phase2_Finetuning_singleGPU_output.ipynb` | 13/13 |
| [07_extended_data_eta.ipynb](07_extended_data_eta.ipynb) | `EdgeVLM_256M_Phase2_Finetuning_with_ETA_output.ipynb` | 8/13 |
| [08_dual_crop_baseline.ipynb](08_dual_crop_baseline.ipynb) | `V2_EdgeVLM_256M_Phase2_All_Techniques_output.ipynb` | 13/13 |
| [09_revised_no_mediapipe.ipynb](09_revised_no_mediapipe.ipynb) | `V3_EdgeVLM_256M_Phase2_Improved_NoMediaPipe_run.ipynb` | 0/13 |
| [10_no_mediapipe_evaluation.ipynb](10_no_mediapipe_evaluation.ipynb) | `V3_EdgeVLM_256M_Phase2_NoMediaPipe_EvaluationOnly_run.ipynb` | 4/5 |
| [11_mediapipe_training.ipynb](11_mediapipe_training.ipynb) | `V3_EdgeVLM_256M_Phase2_MediaPipe_Enhanced_resume16100.ipynb` | 0/14 |
| [12_mediapipe_evaluation.ipynb](12_mediapipe_evaluation.ipynb) | `V3_EdgeVLM_256M_MediaPipe_PostTraining_Results.ipynb` | 10/11 |
| [dual_crop_baseline_source.ipynb](archive/dual_crop_baseline_source.ipynb) | `V2_EdgeVLM_256M_Phase2_All_Techniques.ipynb` | 0/13 |
| [no_mediapipe_source.ipynb](archive/no_mediapipe_source.ipynb) | `V3_EdgeVLM_256M_Phase2_Improved_NoMediaPipe.ipynb` | 0/13 |
| [mediapipe_source.ipynb](archive/mediapipe_source.ipynb) | `V3_EdgeVLM_256M_Phase2_MediaPipe_Enhanced.ipynb` | 0/14 |
| [extended_data_eta_source.ipynb](archive/extended_data_eta_source.ipynb) | `EdgeVLM_256M_Phase2_Finetuning_with_ETA.ipynb` | 0/13 |
| [prompt_benchmark.ipynb](../prompt_engineering/prompt_benchmark.ipynb) | `EdgeVLM_prompting_langgraph_full_test.ipynb` | 4/8 |

## Execution and preservation

The original downloads are unchanged. Publication copies retain code, numeric/text results and execution counts; embedded visual/HTML outputs, widget metadata and image-path previews are removed. Credentials are redacted if found. The manifest records source hashes and output counts. Code still contains original server paths and optional installation/export cells. Review these before running; these are experiment records rather than a portable training CLI.

`support/` contains the downloaded helper versions. The current helper is not assumed identical to the helper used at every historical run. Existing shared 500M pipeline files were not replaced.

Training implementation: notebooks 01-09 and 11. Saved-result scoring: 10 and 12, or `scripts/evaluate_256m_saved.py`. The source-only training notebooks do not become executed records merely because their CSV results exist.
