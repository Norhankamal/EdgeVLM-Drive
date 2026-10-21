# SmolVLM-256M series

The 256M model is the smallest configuration that fits the Jetson Nano without swap (≈ 489 MB package,
724 MB peak GPU memory) and is the preferred on-device model.

| Step | Notebook | Input | Result file |
|---|---|---|---|
| Single-image LoRA, sampling / class weights | 01-05 | scene only | [`results/smolvlm_256m`](../../results/smolvlm_256m) `single_image_*` |
| Extended data | 06-07 | scene only | – |
| Dual-crop baseline (vision + text LoRA) | 08 | scene + face | `dual_crop_baseline.json` |
| Revised protocol, Haar face crops | 09-10 | scene + face | `revised_no_mediapipe.json` |
| MediaPipe-guided crop and cue text | 11-12 | scene + face + cues | `mediapipe.json` |

Macro-F1 on the 5,558-image test set: 78.16 % → 79.63 % → 80.61 %. The three configurations differ in
several settings at once, so the series documents development rather than isolating one component.

- Guides: [experiments](smolvlm256m_experiments.md) · [model loading](smolvlm256m_model_loading.md) ·
  [file index](smolvlm256m_file_index.md) · [experiment log](experiment_log.md)
- Notebooks: [`notebooks/finetuning_smolvlm256`](../../notebooks/finetuning_smolvlm256)
- Score saved predictions without a model: `python scripts/evaluate_256m_saved.py --predictions preds.csv --output metrics.json`
- On-device results: [`results/jetson`](../../results/jetson)
