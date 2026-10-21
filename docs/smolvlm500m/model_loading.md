# SmolVLM 500M model dependencies

No model weights are committed and no quantized model is created.

## Loading chain

Every 500M adapter is trained directly on the pretrained `HuggingFaceTB/SmolVLM-500M-Instruct` base; no earlier
adapter is merged first (unlike the 256M dual-crop chains in [`../smolvlm256m/model_loading.md`](../smolvlm256m/model_loading.md)).
To evaluate: load the base, then `PeftModel.from_pretrained(base, <run>/best_by_macro_f1)` together with the processor
saved in that folder. To deploy: merge that adapter into the base (`merge_and_unload`) and save FP16 safetensors; see
`scripts/export_model.py`.

Use `best_by_macro_f1/` for every reported result. The run folder itself holds the *last-epoch* adapter, which can be
worse, and `checkpoint-*` folders are optimiser-resume state.

## The instruction (prompt) is part of the trained model

A fine-tuned adapter learns to answer one specific instruction. The text used at inference must be the text used
in training; the preprocessing version should match as well.

| Run | Instruction | Preprocessing | Where defined |
|---|---|---|---|
| Initial baseline (notebook 01) | `INSTRUCTION_V1` (short two-image instruction) | v1 | `edgevlm.constants` |
| Extended data v1 (notebook 02) | `INSTRUCTION_V1` | v1 | `edgevlm.constants` |
| Final extended-data run (notebook 04) | `INSTRUCTION_V2` (defines each class; a visible phone decides PHONE_USAGE) | v2 | `edgevlm.constants` |

The hardware pipeline is not part of this repository, so which instruction it sends is not documented here. Before
deploying any model, check the pipeline against the table above:

1. the instruction it sends must be the exact text the adapter was trained with (including the line breaks);
2. its preprocessing must match the version in the table (`src/edgevlm/pipeline/langgraph_pipeline.py` is version 2);
3. re-run the on-device validation with that combination.

Serving an adapter with a different instruction is a train/inference mismatch whose effect on accuracy has not been
measured.

`scripts/export_model.py` requires `--instruction v1|v2` and writes the instruction text, its SHA-256 and its version
to the export folder (`instruction.txt` and `deployment_info.txt`), so the package carries the prompt it expects.

## Size and device notes

The merged FP16 package is about 968 MB and, on the 4 GB Jetson Nano, needs a 4 GB swap partition (interim report,
Sec. VII-D of the nine-page version). GPU execution of the scripts and the export equivalence were not tested in this repository.
