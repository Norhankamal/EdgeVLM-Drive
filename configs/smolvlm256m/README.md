# SmolVLM 256M configuration records

Each JSON extracts uppercase assignments and LoRA/TrainingArguments keyword expressions from its corresponding notebook, with zero-based source cell indices and a source SHA-256. Expressions are recorded without execution, so referenced variables and alternate/smoke-test cells remain visible. Follow the notebook execution order to resolve them.

**These are provenance records, not YAML inputs for `scripts/train.py`.** The existing trainer has different prompt, preprocessing and initialization assumptions. In particular, MediaPipe crop/cue support is implemented in the 256M notebook, not enabled by changing the model ID in a 500M configuration.

No quantized model or quantization configuration is added. The notebooks preserve the original optional FP32 export code; export was not run during this repository update.
