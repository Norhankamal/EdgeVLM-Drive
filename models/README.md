# Model weights

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
