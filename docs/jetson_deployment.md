# Jetson Nano deployment

## Target

Original NVIDIA Jetson Nano 4 GB (JetPack 4.6.6, CUDA 10.2, PyTorch 1.13.0, Python 3.8.10). The
operating system, ROS 2, Python, model weights and activations share one 4 GB memory pool, and the
Maxwell GPU cannot move to CUDA 11 or newer.

## Deployed package

The selected LoRA adapter is merged into the base model (`merge_and_unload`) and saved as a
standalone FP16 PyTorch / Hugging Face model, so PEFT is not needed at runtime.

| Model | Package | Swap | Peak GPU memory | Mean latency |
|---|---:|---|---:|---:|
| SmolVLM-256M | ≈ 489 MB | not needed | 724 MB | 6.8 s |
| SmolVLM-500M | ≈ 968 MB | 4 GB swap partition | 1,217 MB | 7.9 s |

Each deployed model is served with the instruction it was trained with: the DMD-data and
extended-data models with the shorter instruction (`INSTRUCTION_V1`), the refined-protocol model
with the extended instruction (`INSTRUCTION_V2`). Set the pipeline instruction to match the model.

## Runtime selection

| Route | Outcome on this platform |
|---|---|
| **PyTorch FP16, merged package** | **works** (deployed) |
| ONNX Runtime | Python 3.6 vs Transformers, unsupported fused-decoder operators in ONNX Runtime 1.10 (needs an unfused opset-14 export), then a ConvInteger failure in the INT8 vision encoder, which also agreed with the source model on only 80 % of top-1 predictions |
| LFM2-VL-450M (GGUF, ≈ 219 MB) | needs llama.cpp, whose GPU build requires CUDA 11.0 or newer |
| Moondream | no verified standalone 0.5B artifact; the 2B model needs ≈ 3.85 GB before inference |
| BitsAndBytes, optimum-quanto, ONNX Runtime GPU, TensorRT multimodal conversion, CPU dynamic quantisation | blocked by CUDA, PyTorch, ARM64 or runtime limits |

## Measuring on the device

```bash
python scripts/benchmark_jetson.py --package export/smolvlm_256m --csv test.csv \
       --data-root /path/to/data --per-class 300 --instruction v1 --out jetson_256m.json
```

Use the 5 W power mode if the power supply cannot sustain the maximum mode, and keep the same mode
when comparing models. Reported results: [`results/jetson`](../results/jetson).
