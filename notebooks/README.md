# Notebooks

Executed experiment records. Each notebook starts with a header table (model, stage, purpose, inputs,
outputs, configuration, results file). Set the paths in the first configuration cell to your data and
split folders before running.

| Folder | Content |
|---|---|
| [`finetuning_smolvlm500`](finetuning_smolvlm500) | SmolVLM-500M: Phase 1 baseline and three Phase 2 runs on the extended data |
| [`finetuning_smolvlm256`](finetuning_smolvlm256) | SmolVLM-256M: 12-step development sequence from single-image LoRA to the MediaPipe-guided model |
| [`prompt_engineering`](prompt_engineering) | instruction-only prompt benchmark through the LangGraph pipeline |

The command-line equivalents for the 500M runs are `scripts/train.py`, `scripts/evaluate.py` and
`scripts/export_model.py`; saved 256M predictions are scored with `scripts/evaluate_256m_saved.py`.
