# SmolVLM 500M file index

Paths are relative to the repository root. The 256M files are indexed in [`../smolvlm256m/file_index.md`](../smolvlm256m/file_index.md).

## Notebooks (experiment records)

| File | Purpose |
|---|---|
| [`notebooks/finetuning_smolvlm500/README.md`](../../notebooks/finetuning_smolvlm500/README.md) | Notebook guide: original-file mapping, saved execution status, preservation notes. |
| [`notebooks/finetuning_smolvlm500/source_manifest.json`](../../notebooks/finetuning_smolvlm500/source_manifest.json) | Source and repository SHA-256, executed-cell counts, removed outputs and edits per notebook. |
| [`notebooks/finetuning_smolvlm500/01_phase1_baseline_dualcrop_lora.ipynb`](../../notebooks/finetuning_smolvlm500/01_phase1_baseline_dualcrop_lora.ipynb) | Initial baseline reported in the interim report. |
| [`notebooks/finetuning_smolvlm500/02_phase2_extended_data_v1.ipynb`](../../notebooks/finetuning_smolvlm500/02_phase2_extended_data_v1.ipynb) | Extended data, LoRA r = 16. |
| [`notebooks/finetuning_smolvlm500/03_phase2_extended_data_v2.ipynb`](../../notebooks/finetuning_smolvlm500/03_phase2_extended_data_v2.ipynb) | Code snapshot; saved outputs identical to notebook 02. |
| [`notebooks/finetuning_smolvlm500/04_phase2_extended_data_final.ipynb`](../../notebooks/finetuning_smolvlm500/04_phase2_extended_data_final.ipynb) | Final extended-data run (preprocessing v2, extended instruction). |

## Configurations

| File | Purpose |
|---|---|
| [`configs/smolvlm500m/README.md`](../../configs/smolvlm500m/README.md) | Which YAML reproduces which notebook run. |
| [`configs/smolvlm500m/phase1_baseline.yaml`](../../configs/smolvlm500m/phase1_baseline.yaml) | Initial baseline settings (r = 8, 160-image check, preprocessing v1, `INSTRUCTION_V1`). |
| [`configs/smolvlm500m/phase2_v1.yaml`](../../configs/smolvlm500m/phase2_v1.yaml) | Extended-data v1 settings (r = 16, 320-image check). |
| [`configs/smolvlm500m/phase2_final.yaml`](../../configs/smolvlm500m/phase2_final.yaml) | Final run settings (preprocessing v2, `INSTRUCTION_V2`, damped sampler, 600-image check). |

## Results

| File | Purpose |
|---|---|
| [`results/smolvlm_500m/README.md`](../../results/smolvlm_500m/README.md) | Guide to the stored metrics and how they were extracted. |
| [`results/smolvlm_500m/source_manifest.json`](../../results/smolvlm_500m/source_manifest.json) | Source notebook, original-file hash, extraction and verification per result file. |
| [`results/smolvlm_500m/phase1_baseline.json`](../../results/smolvlm_500m/phase1_baseline.json) | Test metrics, per-class table, confusion matrix and epoch history of the initial baseline. |
| [`results/smolvlm_500m/phase2_v1.json`](../../results/smolvlm_500m/phase2_v1.json) | Same, extended-data v1. |
| [`results/smolvlm_500m/phase2_final.json`](../../results/smolvlm_500m/phase2_final.json) | Same, final run. |
| [`results/smolvlm_500m/jetson_validation_phase1.json`](../../results/smolvlm_500m/jetson_validation_phase1.json) | On-device results of the initial baseline model, transcribed from the interim report. |

## Documentation

| File | Purpose |
|---|---|
| [`docs/smolvlm500m/experiments.md`](../../docs/smolvlm500m/experiments.md) | Experiment progression with the key numbers. |
| [`docs/smolvlm500m/experiment_log.md`](../../docs/smolvlm500m/experiment_log.md) | Run lineage, differences between runs, open questions. |
| [`docs/smolvlm500m/model_loading.md`](../../docs/smolvlm500m/model_loading.md) | Loading chain, instruction/preprocessing alignment, export notes. |
| [`docs/smolvlm500m/file_index.md`](../../docs/smolvlm500m/file_index.md) | This file. |
| [`docs/reports/README.md`](../../docs/reports/README.md) | Why the interim report is kept out of Git by default. |

## Code and scripts

| File | Purpose |
|---|---|
| [`scripts/train.py`](../../scripts/train.py) | Fine-tune from a YAML config (macro-F1 checkpoint selection, resumable). |
| [`scripts/evaluate.py`](../../scripts/evaluate.py) | Score an adapter on a split, one image at a time. |
| [`scripts/export_model.py`](../../scripts/export_model.py) | Merge the adapter and export FP16; requires `--instruction v1|v2` and records it in the package. |
| [`src/edgevlm/constants.py`](../../src/edgevlm/constants.py) | Labels and the two instruction texts (verbatim from the notebooks). |
| [`src/edgevlm/config.py`](../../src/edgevlm/config.py) | YAML to typed configuration. |
| [`src/edgevlm/pipeline/langgraph_pipeline.py`](../../src/edgevlm/pipeline/langgraph_pipeline.py) | Current pipeline and preprocessing (v2). |
| [`src/edgevlm/pipeline/legacy/langgraph_pipeline_v1.py`](../../src/edgevlm/pipeline/legacy/langgraph_pipeline_v1.py) | Preprocessing used by notebooks 01 and 02 (v1). |
| [`src/edgevlm/data/dataset.py`](../../src/edgevlm/data/dataset.py) | Dual-crop dataset and collator. |
| [`src/edgevlm/data/frame_paths.py`](../../src/edgevlm/data/frame_paths.py) | Frame path resolver. |
| [`src/edgevlm/modeling/lora.py`](../../src/edgevlm/modeling/lora.py) | Vision + text LoRA. |
| [`src/edgevlm/training/callbacks.py`](../../src/edgevlm/training/callbacks.py) | Macro-F1 early stopping and run monitor. |
| [`src/edgevlm/training/trainer.py`](../../src/edgevlm/training/trainer.py) | Balanced sampler and `TrainingArguments`. |
| [`src/edgevlm/training/resume.py`](../../src/edgevlm/training/resume.py) | Checkpoint recovery. |
| [`src/edgevlm/evaluation/metrics.py`](../../src/edgevlm/evaluation/metrics.py) | Parsing and metric definitions. |
| [`src/edgevlm/evaluation/inference.py`](../../src/edgevlm/evaluation/inference.py) | Single and batched dual-crop inference. |
| [`src/edgevlm/deployment/export.py`](../../src/edgevlm/deployment/export.py) | Adapter merge and FP16 export; records the training instruction. |

## Tests

| File | Purpose |
|---|---|
| [`tests/test_metrics.py`](../../tests/test_metrics.py) | Metric code reproduces the published confusion-matrix numbers. |
| [`tests/test_results_consistency.py`](../../tests/test_results_consistency.py) | Stored metrics follow from the stored confusion matrices. |
| [`tests/test_constants_and_notebooks.py`](../../tests/test_constants_and_notebooks.py) | Prompts equal the notebooks'; notebooks valid and free of images and credentials. |
| [`tests/test_smolvlm500_manifest.py`](../../tests/test_smolvlm500_manifest.py) | Manifest hashes match the notebooks in the repository. |
| [`tests/test_config.py`](../../tests/test_config.py) | Every config loads and carries the documented settings. |
| [`tests/test_export_info.py`](../../tests/test_export_info.py) | Export requires and records the instruction. |
| [`tests/test_training_logic.py`](../../tests/test_training_logic.py) | Sampler weights, LoRA targets, early stopping, resume. |
| [`tests/test_frame_paths.py`](../../tests/test_frame_paths.py) | Frame path resolution. |
| [`tests/test_preprocessing.py`](../../tests/test_preprocessing.py) | Both preprocessing versions on a synthetic frame. |
| [`tests/test_docs_links.py`](../../tests/test_docs_links.py) | Every relative Markdown link resolves. |

## Changes to existing files in this update

- `README.md`: restructured so both models are documented symmetrically; adds the 500M prompt-alignment note and a 500M versus 256M comparison on the shared test set.
- `models/README.md`: export command now passes `--instruction`; links to the relocated 256M notes.
- `docs/smolvlm256m/`: the 256M documents moved here from a misspelled folder and links fixed (see its file index).
- `configs/`: the 500M YAML files moved into `configs/smolvlm500m/`.
- `src/edgevlm/deployment/export.py`, `scripts/export_model.py`: the export requires and records the instruction version.
- `tests/`: notebook and config tests point at the current folders; three new test files.

No notebook, result value or model was changed. No GPU run, export or Jetson benchmark was performed.
