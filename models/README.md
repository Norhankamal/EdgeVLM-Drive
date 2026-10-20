# Model weights

## SmolVLM 500M

For the 500M experiments, fine-tuning first produces a LoRA adapter. The selected adapter is
then merged into the base model and exported as the standalone package described below.

Weights are **not** stored in Git (the merged FP16 package is ~968 MB, above GitHub's 100 MB file
limit). Produce them with the export script, or download them once they are published.

| Artefact | Content | Size | Where |
|---|---|---:|---|
| LoRA adapter | `best_by_macro_f1/` (`adapter_model.safetensors`, `adapter_config.json`, processor files) | a few MB | produced by `scripts/train.py` |
| Merged deployment package | base model + adapter merged, FP16 safetensors, tokenizer/processor, `deployment_info.txt` | ~968 MB | produced by `scripts/export_model.py` |

Public download link: *to be added once the package is uploaded (for example to the Hugging Face Hub).*

```bash
python scripts/export_model.py --adapter runs/<run>/best_by_macro_f1 --out export/smolvlm_500m --zip
```

The package loads with plain Transformers (`AutoModelForVision2Seq.from_pretrained(path)`), needs no
PEFT and no ONNX. On a 4 GB Jetson Nano the 500M model requires a 4 GB swap partition.
Base model: [`HuggingFaceTB/SmolVLM-500M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)
— check its licence before redistributing derived weights.

## SmolVLM 256M

Base model: `HuggingFaceTB/SmolVLM-256M-Instruct`. The experiments progress from initial
language-decoder adaptation to vision-and-text adaptation with paired inputs and, later,
MediaPipe-guided crops and facial cues. Loading must reproduce the starting weights of the
specific experiment; an adapter is not a standalone full model.

### Loading order by configuration

| Configuration | Required loading order |
|---|---|
| Initial single-image experiment | Pretrained instruction base, then that experiment's language-decoder adapter |
| Dual-crop baseline | Pretrained base, merge the exact earlier adapter used for initialization, then load the selected vision-and-text adapter |
| Revised No-MediaPipe | Pretrained instruction base, then the selected revised adapter; do not add the earlier initialization adapter |
| MediaPipe crop and cues | Pretrained base, merge the exact earlier initialization adapter, then load the selected MediaPipe experiment adapter |

The baseline and MediaPipe loading chains must not be replaced by loading only their final
adapter on an untouched base. Preserve the identity and hash of the earlier adapter rather
than relying on a mutable folder name. The original notebook code records how that adapter
was selected; the current exports do not independently prove every historical weight version.

### Files and preprocessing to preserve

- Keep `adapter_model.safetensors` and `adapter_config.json` from the selected experiment.
- Keep its matching processor/tokenizer files and chat template, including
  `preprocessor_config.json`, `processor_config.json`, `tokenizer_config.json`, tokenizer
  vocabulary files and special-token settings where present.
- Preserve the exact prompt and full-scene/face-crop preprocessing. For MediaPipe, also retain
  the face-landmarker task version and matching crop/cue cache or reproducible extraction setup.
- Use the selected `best_by_macro_f1` adapter for the reported selected-model workflow.
  `last_adapter` holds the terminal model and may differ; optimizer-resume checkpoints serve
  a separate purpose.

Use [baseline notebook 08](../notebooks/finetuning_smolvlm256/08_dual_crop_baseline.ipynb),
[revised notebook 09](../notebooks/finetuning_smolvlm256/09_revised_no_mediapipe.ipynb) and
[MediaPipe notebook 11](../notebooks/finetuning_smolvlm256/11_mediapipe_training.ipynb) for the
configuration-specific initialization and preprocessing code. Historical server paths need
review before execution.

### Availability and export status

Weights are not committed, and no public 256M weight-download URL is supplied here. The
saved-prediction scorer does not need weights. No quantization is added.

The optional FP32 export code in the notebooks is distinct from the existing shared export CLI.
That CLI is not verified for the chained 256M initialization described above. No new export
or inference-equivalence test was run during repository preparation; no 256M package size
or Jetson performance is inferred from the other model's measurements.

See the [256M model dependency notes](../docs/smalvlm256/smolvlm256m_model_loading.md) and
[results guide](../results/smolvlm_256m/README.md) for provenance and evaluation details.
