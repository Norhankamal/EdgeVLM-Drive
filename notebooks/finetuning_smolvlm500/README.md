# SmolVLM 500M notebooks

Read notebooks 01 through 04 as the development sequence: initial baseline, extended-data v1, a code
snapshot (v2) and the final extended-data run. No cells were executed during repository preparation.

| Repository notebook | Original download | Saved executed code cells |
|---|---|---:|
| [01_phase1_baseline_dualcrop_lora.ipynb](01_phase1_baseline_dualcrop_lora.ipynb) | `EdgeVLM_finetuning_smolvlm500m_dualcrop_V1Phase1.ipynb` | 29/30 |
| [02_phase2_extended_data_v1.ipynb](02_phase2_extended_data_v1.ipynb) | `EdgeVLM_finetuning_smolvlm500m_dualcrop_Phase2V1.ipynb` | 29/33 |
| [03_phase2_extended_data_v2.ipynb](03_phase2_extended_data_v2.ipynb) | `EdgeVLM_finetuning_smolvlm500m_dualcrop_Phase2V2.ipynb` | 29/33 |
| [04_phase2_extended_data_final.ipynb](04_phase2_extended_data_final.ipynb) | `EdgeVLM_finetuning_smolvlm500m_dualcrop_Phase2_Final.ipynb` | 29/33 |

`EdgeVLM_finetuning_smolvlm500m_dualcrop_Phase2_Final_1.ipynb` is a byte-identical re-download of the
final notebook and is not added separately.

## What each notebook is

| Notebook | Role | Test result (saved output) |
|---|---|---|
| 01 | Initial baseline, reported in the interim report | macro-F1 74.13 % on 3,579 images |
| 02 | Extended data, LoRA r = 16 | macro-F1 79.56 % on 5,558 images |
| 03 | Code snapshot; outputs identical to 02 | none of its own |
| 04 | Final run: preprocessing v2, extended instruction, damped sampler | macro-F1 80.18 % on 5,558 images |

The extended-data test set differs from the initial baseline's, so compare runs only within the same
test set. How the runs differ and what remains open: [experiment log](../../docs/smolvlm500m/experiment_log.md).

## Execution and preservation

The original downloads are unchanged. Publication copies keep every code cell, the numeric/text outputs
and the execution counts; the driver-frame preview image (one per notebook), widget metadata and empty
trailing cells are removed. The executable code in each copy is identical to the original (checked after
removing comments and docstrings); comments or docstrings were edited in 3, 1, 1 and 1 code cells of
notebooks 01 to 04, and the title cell and some explanatory markdown were reworded. No credentials were found.
[`source_manifest.json`](source_manifest.json) records source and repository hashes, execution counts and
these edits.

Code cells still contain the original server paths (`/home/jovyan/...`) and optional installation and export
cells; these are experiment records, not a portable training CLI. For a parameterised entry point use
[`scripts/train.py`](../../scripts/train.py) with [`configs/smolvlm500m/`](../../configs/smolvlm500m/README.md).

The shared pipeline helpers live in `src/edgevlm/pipeline/` (current) and `src/edgevlm/pipeline/legacy/`
(version used by notebooks 01 and 02); they are the files the notebooks import as `langgraph_pipeline`.
The helper used at each historical run is identified by the experiment table in the experiment log.
