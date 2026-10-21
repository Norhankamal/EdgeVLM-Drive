# SmolVLM 500M training configurations

These YAML files are inputs to [`scripts/train.py`](../../scripts/train.py) and
[`scripts/evaluate.py`](../../scripts/evaluate.py). They encode the settings documented for each notebook run:

| Config | Reproduces | Notebook |
|---|---|---|
| [`phase1_baseline.yaml`](phase1_baseline.yaml) | initial baseline: r = 8, 160-image check, full inverse-frequency sampler, preprocessing v1, `INSTRUCTION_V1` | [01](../../notebooks/finetuning_smolvlm500/01_phase1_baseline_dualcrop_lora.ipynb) |
| [`phase2_v1.yaml`](phase2_v1.yaml) | extended data v1: r = 16, 320-image check, preprocessing v1, `INSTRUCTION_V1` | [02](../../notebooks/finetuning_smolvlm500/02_phase2_extended_data_v1.ipynb) |
| [`phase2_final.yaml`](phase2_final.yaml) | final run: preprocessing v2, `INSTRUCTION_V2`, damped sampler, 600-image check | [04](../../notebooks/finetuning_smolvlm500/04_phase2_extended_data_final.ipynb) |

The run identifiers keep their original `phase*` names so that they match the result files. Machine-specific
locations are not stored: set `EDGEVLM_DATA_ROOT` (see the root README). Tests check that each file loads and carries
the documented hyper-parameters; the scripts themselves have not been executed end-to-end on a GPU, and the notebooks
remain the executed record. For the 256M configuration records (static JSON provenance, not runnable inputs) see
[`../smolvlm256m/README.md`](../smolvlm256m/README.md).
