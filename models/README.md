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

The 256M experiments progress from initial adaptation through revised training to the
MediaPipe crop-and-cue configuration. When a later experiment starts from an earlier adapted
model, that earlier-adapter initialization must be preserved when loading its weights.
See [256M model dependencies](../docs/smolvlm256m_model_loading.md) for the loading requirements
of each configuration. Weights remain outside Git. No quantization is added.
