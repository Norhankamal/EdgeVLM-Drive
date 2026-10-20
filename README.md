# EdgeVLM-Drive

**Resource-aware vision-language driver monitoring on NVIDIA Jetson Nano**

[![CI](https://github.com/Norhankamal/EdgeVLM-Drive/actions/workflows/ci.yml/badge.svg)](https://github.com/Norhankamal/EdgeVLM-Drive/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

EdgeVLM-Drive classifies a driver's state from a single in-cabin camera frame into
**NORMAL**, **DROWSY**, **LOOKING_AWAY** or **PHONE_USAGE** using a compact vision-language
model (SmolVLM) that is small enough to run on a 4 GB Jetson Nano. Everything runs on the
device: no cloud round-trip, no driver-facing video leaving the cabin.

> **Project status.** This repository contains the **SmolVLM-500M fine-tuning** work (training, evaluation, export and experiment records) and the **SmolVLM-256M experiment notebooks, configuration records and saved-result evaluation**. Data-preparation, prompt-engineering and Jetson/ROS integration contributions are still being added; see [Roadmap](#roadmap). The 256M section distinguishes historical notebooks from newly tested scoring code.

---

## Contents

- [Approach](#approach)
- [Results](#results)
- [Limitations](#limitations)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Data and model weights](#data-and-model-weights)
- [Documentation](#documentation)
- [Roadmap](#roadmap)
- [Authors](#authors) · [Acknowledgements](#acknowledgements) · [License](#license)

## Approach

Fine-tuning `HuggingFaceTB/SmolVLM-500M-Instruct` rests on three design choices:

1. **Dual-crop input.** Each frame becomes two images: a 448×448 *full-scene* crop (hands, lap,
   phone context) and a 256×256 *face* crop (eyelids, head pose). Training reuses the exact
   preprocessing functions of the deployed pipeline, so the model is trained on the pixels it will
   see at inference.
2. **LoRA on the vision encoder *and* the text decoder** (attention projections, r = 8 in the
   initial baseline and r = 16 in the later extended-data experiments). Adapting the vision encoder lets the model learn a *new visual
   cue* such as eye closure, instead of only re-mapping cues it could already see.
3. **Checkpoint selection by validation macro-F1, not `eval_loss`.** With heavy class imbalance a
   model can reach a low loss by favouring NORMAL while DROWSY recall collapses. A callback runs
   real `generate()` calls on a fixed, class-balanced validation subset after every epoch, keeps the
   best macro-F1 adapter and early-stops on it.

The best adapter is merged into the base weights and exported as a standalone **FP16 PyTorch /
safetensors** package (no PEFT at runtime, **no ONNX** — the ONNX route was abandoned because of
operator incompatibilities on JetPack 4.6.6; see the interim report, Sec. VI-D).

The surrounding system is a LangGraph state graph (preprocess → decision gate → VLM inference →
parse/validate → temporal decision → alert). Fine-tuning evaluation here calls the model directly;
it does not exercise the graph's retry, temporal voting or alerting.

## Results

All numbers below come from the executed notebooks or the interim report and are stored in
[`results/`](results/smolvlm_500m). Accuracy alone is misleading here (NORMAL is ~60 % of the test
sets), so balanced accuracy, macro-F1 and DROWSY recall are given as well.

### Initial baseline (reported in the interim report)

We first established the 500M dual-crop baseline and evaluated it on the original test set.

Test set: **3,579 images**, subject-disjoint split of DMD. GPU-server evaluation, 100 % valid label outputs.

| Accuracy | Balanced acc. | Macro-F1 | Unsafe-state miss rate | False-alarm rate |
|---:|---:|---:|---:|---:|
| 85.14 % | 74.84 % | 74.13 % | 27.66 % | 6.37 % |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NORMAL | 0.879 | 0.936 | 0.907 | 2,151 |
| DROWSY | 0.539 | 0.536 | 0.537 | 168 |
| LOOKING_AWAY | 0.608 | 0.785 | 0.685 | 302 |
| PHONE_USAGE | 0.966 | 0.737 | 0.836 | 958 |

### Validation of the baseline on Jetson Nano 4 GB

We then evaluated the baseline model on the target device using a smaller diagnostic subset.

Merged FP16 SmolVLM-500M (~968 MB), JetPack 4.6.6, PyTorch 1.13.0; the 500M model needs a 4 GB swap
partition. Measured on a **120-image balanced diagnostic subset** (30 per class), so accuracy is not
comparable to the full-test numbers.

| Accuracy | Mean latency | P95 latency | DROWSY recall | DROWSY → NORMAL |
|---:|---:|---:|---:|---:|
| 80.00 % | 4,659 ms | 5,623 ms | 36.67 % (11/30) | 19/30 |

### Expanded dataset and revised training (not in the interim report)

Next, we extended the dataset and increased the LoRA rank. A subsequent experiment revised
preprocessing, the prompt, sampling strength and the validation check.

Test set: **5,558 images** — a *different, larger* test set than the initial baseline (1,021 DROWSY frames
instead of 168), so **these numbers are not directly comparable to the initial baseline results**.

| Run | Changes vs. previous | Macro-F1 | Balanced acc. | DROWSY recall | False-alarm rate |
|---|---|---:|---:|---:|---:|
| `phase2_v1` | extended data, LoRA r = 16 | 79.56 % | 83.00 % | 81.3 % | 10.22 % |
| `phase2_final` | + current preprocessing, extended prompt, damped sampler, larger validation check | 80.18 % | 80.08 % | 80.9 % | 7.75 % |

The original run identifiers in the table are retained in filenames and commands for reproducibility.

Several factors changed at once between the initial baseline, `phase2_v1` and `phase2_final`, and the final prompt
was refined after looking at a test-set confusion matrix, so the improvements cannot be attributed to
a single change and should be read as optimistic. Details and open questions:
[`docs/experiment_log.md`](docs/experiment_log.md). The later extended-data models have not been run on the Jetson.

## Limitations

- **DROWSY frames are often classified as NORMAL** (on the Jetson diagnostic subset 19 of 30) — a
  safety-critical error, since a NORMAL prediction raises no alert. Single frames cannot separate
  sustained eye closure from a blink; temporal modelling over short sequences is future work.
- DROWSY labels are derived from DMD eye-state annotations (a ≥ 0.8 s closure episode); the class is
  rare and drawn from few subjects, so subject/lighting diversity is limited.
- Reported latency is single-image `generate()` on a GPU server, or on the Jetson for the 120-image
  subset. Memory, quantisation and the later extended-data models on-device remain to be measured.
- Single runs, no confidence intervals or statistical tests.
- The `scripts/` entry points are a refactor of the notebooks and have **not yet been executed
  end-to-end on a GPU**; the notebooks are the executed record. Unit tests cover the logic that does
  not need a GPU (see [Tests](#tests)).

## Repository structure

```
EdgeVLM-Drive/
├── configs/                  # one YAML per documented training run
├── data/                     # (not tracked) dataset notes only - see data/README.md
├── demo/                     # demo video link and material
├── docs/
│   ├── reports/              # interim report (local copy; see docs/reports/README.md)
│   └── experiment_log.md     # run lineage, differences between runs, open questions
├── models/                   # (not tracked) weights notes only - see models/README.md
├── notebooks/
│   └── 04_finetuning_smolvlm_500m/   # executed experiment records (outputs kept, images removed)
├── results/smolvlm_500m/     # metrics, confusion matrices, epoch histories (JSON)
├── scripts/                  # train.py · evaluate.py · export_model.py
├── src/edgevlm/
│   ├── constants.py          # labels, prompts (verbatim from the notebooks)
│   ├── config.py             # YAML -> typed config
│   ├── pipeline/             # LangGraph pipeline + dual-crop preprocessing (v2; legacy/ = v1)
│   ├── data/                 # frame path resolver, dual-crop dataset, collator
│   ├── modeling/             # vision+text LoRA
│   ├── training/             # macro-F1 early stopping, balanced trainer, resume logic
│   ├── evaluation/           # metrics and inference
│   └── deployment/           # adapter merge + FP16 export
└── tests/
```

Notebook folders are numbered by project stage (`01` data preparation, `02` prompt engineering,
`03` SmolVLM-256M, `04` SmolVLM-500M, `05` Jetson deployment); `04` is the first one populated.

## Getting started

```bash
git clone https://github.com/Norhankamal/EdgeVLM-Drive.git
cd EdgeVLM-Drive
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -e ".[dev]"            # package + tests (no GPU stack)
pip install -e ".[train]"          # add torch / transformers / peft for training and evaluation
```

Install the PyTorch build that matches your CUDA version from pytorch.org first if you need GPU
support. The reported runs used Python 3.12, PyTorch 2.11 (CUDA 12.6), Transformers 4.46.3,
PEFT 0.21 and an NVIDIA A16.

### Train, evaluate, export

The datasets are **not** included ([why](#data-and-model-weights)). Point `EDGEVLM_DATA_ROOT` at the
directory the CSV `frame_path` columns are relative to, then:

```bash
export EDGEVLM_DATA_ROOT=/path/to/data

python scripts/train.py    --config configs/smolvlm500m_phase2_final.yaml --smoke-test   # 3 steps first
python scripts/train.py    --config configs/smolvlm500m_phase2_final.yaml
python scripts/evaluate.py --config configs/smolvlm500m_phase2_final.yaml \
       --adapter runs/smolvlm500m_phase2_final/best_by_macro_f1 --split test
python scripts/export_model.py --adapter runs/smolvlm500m_phase2_final/best_by_macro_f1 \
       --out export/smolvlm_500m
```

`configs/smolvlm500m_phase1_baseline.yaml` reproduces the settings of the reported baseline
(r = 8, 160-image validation check, full inverse-frequency sampler, legacy preprocessing).
Training was resumable on the original shared server: re-running `train.py` continues from the
newest readable checkpoint (`--no-resume` to start fresh).

### Tests

```bash
pytest
```

The suite checks, among other things, that the metric code reproduces the published initial baseline numbers
(74.13 % macro-F1, 27.66 % miss rate, 6.37 % false-alarm rate) from the confusion matrix, that the
prompts equal the ones in the notebooks, and that the notebooks contain no embedded images or
credentials. GPU-dependent code (model loading, `Trainer`, export) is not covered.

## Data and model weights

- **Data.** Frames come from the Driver Monitoring Dataset (DMD; Ortega et al., ECCV Workshops 2020)
  and are *not* redistributed here: they show identifiable drivers and are covered by the dataset's
  own terms. See [`data/README.md`](data/README.md) for the expected CSV format and the labelling
  rules used.
- **Weights.** Model weights are not stored in Git (the merged FP16 package is ~968 MB).
  See [`models/README.md`](models/README.md).

## Documentation

- Interim report (`docs/reports/`, not committed by default — see [`docs/reports/README.md`](docs/reports/README.md)) — full method, data pipeline, SmolVLM-256M and prompting experiments, Jetson deployment analysis
- [Experiment log](docs/experiment_log.md) — how the four notebooks relate, what differs between runs, open questions
- [Results](results/smolvlm_500m/README.md) — stored metrics and how they were extracted

## Roadmap

| Part | Status |
|---|---|
| SmolVLM-500M fine-tuning, evaluation, export | in this repository |
| Data preparation and labelling pipeline (DMD) | to be added |
| Prompt-engineering experiments | to be added |
| SmolVLM-256M experiment notebooks and saved-result evaluation | added; see the 256M contribution below |
| Jetson benchmark scripts and ROS integration | to be added |
| MediaPipe landmark gating to reduce VLM calls | planned (not tested) |
| Demo video | to be added (`demo/`) |

## Authors

Sherouk Mohammed Mahmoud Rashed · Norhan Kamal Hosny Mahmoud · Nehal Mohamed Abd El Ghani ·
Salma Essam Mohamed Hamad
Artificial Intelligence Program, School of Computing, Queen's University — Supervisor: Prof. Hazem Abbas

## Acknowledgements

[SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct) (Hugging Face), the
Driver Monitoring Dataset (DMD; Ortega et al., ECCV Workshops 2020), and the open-source PEFT,
Transformers and LangGraph projects.

## License

Code: [MIT](LICENSE). The DMD dataset and the SmolVLM weights are governed by their own licences.

## SmolVLM 256M contribution

This contribution develops `HuggingFaceTB/SmolVLM-256M-Instruct` for NORMAL, DROWSY,
LOOKING_AWAY and PHONE_USAGE classification. It covers model preparation, fine-tuning and
saved-prediction evaluation; dataset preparation belongs to a separate project contribution.

### Development sequence and techniques

We first adapted the language decoder with LoRA and investigated label formatting. We then
tested augmented training data, inverse-frequency sampling and a separate class-weighted-loss
alternative. Single-image LangGraph evaluation produced valid labels but continued to miss
many drowsy examples. This motivated paired scene and face inputs, adaptation of both model
branches, revised training, and finally a MediaPipe crop-and-cue experiment.

- **Paired inputs:** the full-scene view retains hands, phone and cabin context; the face crop
  provides eye and head detail. The later notebooks reuse preprocessing helpers for training
  and inference. No-MediaPipe uses Haar-based face extraction; the MediaPipe experiment uses
  landmarks to guide the face box and retains fallback behavior.
- **Efficient adaptation:** the initial experiments use text-attention LoRA with rank 8 and
  alpha 16. The later dual-crop models use vision-and-text attention LoRA with rank 16,
  alpha 32 and dropout 0.05. Their training uses FP16, gradient checkpointing, batch size 1,
  eight accumulation steps, AdamW, cosine scheduling, 5% warm-up and weight decay 0.01.
  Prompt and padding tokens are masked so the loss targets the answer tokens.
- **Prompt design:** revised label instructions require visible phone evidence for PHONE_USAGE
  rather than inferring it from gaze alone. The two views provide complementary evidence.
- **Imbalance and augmentation:** the revised runs use square-root inverse-frequency sampling.
  Mild brightness/contrast augmentation is applied during training to LOOKING_AWAY in the
  revised No-MediaPipe run, and to DROWSY and LOOKING_AWAY in the MediaPipe run. Validation
  and test images are not augmented.
- **Model selection:** generated-label validation macro F1 selects the best adapter, with
  early stopping and checkpoint recovery. Maximum epochs are limits, not completed-epoch claims.
- **MediaPipe cues:** cached facial blendshape scores supply blink, gaze and jaw-opening text
  hints in addition to guiding the face crop. This experiment uses MediaPipe during input
  preparation, not as an inference gate that decides whether to call the VLM.

| Later configuration | Starting weights | Learning rate | Validation examples per class | Patience / maximum epochs |
|---|---|---:|---:|---:|
| Dual-crop baseline | Base with an earlier adapter merged | 1e-4 | Up to 80 | 2 / 8 |
| Revised No-MediaPipe | Pretrained instruction base | 5e-5 | Up to 150 | 2 / 6 |
| MediaPipe crop and cues | Base with an earlier adapter merged | 1e-4 | Up to 150 | 3 / 8 |

### Results on the expanded test population

The following scores were recalculated from the saved prediction exports, each containing
**5,558 images**. They are not new training or inference runs.

| Experiment | Accuracy % | Balanced accuracy % | Macro F1 % | DROWSY recall % |
|---|---:|---:|---:|---:|
| Dual-crop baseline | 83.68 | 82.80 | 78.16 | 87.07 |
| Revised No-MediaPipe | 85.16 | 81.89 | 79.63 | 86.88 |
| MediaPipe crop and cues | 85.98 | 82.46 | 80.61 | 85.01 |

Higher aggregate scores do not mean fewer errors in every category. On this full population,
unsafe examples predicted NORMAL number 254 for the baseline, 305 for revised No-MediaPipe
and 316 for MediaPipe. The [256M results guide](results/smolvlm_256m/README.md) provides metric
definitions and links to the full per-class results and confusion matrices in the JSON records.

### Comparison on shared test paths

To reduce the effect of the expanded test population, we also scored the later predictions on
the **3,579 paths shared with the earlier graph exports**, with matching true labels.

| Experiment | Accuracy % | Balanced accuracy % | Macro F1 % | DROWSY recall % |
|---|---:|---:|---:|---:|
| Earlier single-image sampling graph | 79.46 | 48.81 | 50.78 | 0.00 |
| Earlier single-image weighted-loss graph | 78.90 | 58.90 | 55.51 | 2.38 |
| Dual-crop baseline | 84.07 | 78.01 | 73.08 | 63.10 |
| Revised No-MediaPipe | 84.94 | 75.13 | 73.43 | 55.36 |
| MediaPipe crop and cues | 86.31 | 75.08 | 74.87 | 50.60 |

The expanded population adds 1,126 NORMAL and 853 DROWSY paths. Drowsiness recall on the
shared population is lower than on the full later population. Matching paths and labels does
not verify unchanged image contents, and training, initialization and prompts differ between
runs. These results describe development, not a controlled measurement of MediaPipe's effect.

### Finding the notebooks and evaluating saved predictions

The [notebook guide](notebooks/finetuning_smolvlm256/README.md) maps repository copies to their
original filenames and records saved execution status.

| Task | Notebook |
|---|---|
| Dual-crop baseline | [08_dual_crop_baseline.ipynb](notebooks/finetuning_smolvlm256/08_dual_crop_baseline.ipynb) |
| Revised No-MediaPipe training | [09_revised_no_mediapipe.ipynb](notebooks/finetuning_smolvlm256/09_revised_no_mediapipe.ipynb) |
| No-MediaPipe result display | [10_no_mediapipe_evaluation.ipynb](notebooks/finetuning_smolvlm256/10_no_mediapipe_evaluation.ipynb) |
| MediaPipe training | [11_mediapipe_training.ipynb](notebooks/finetuning_smolvlm256/11_mediapipe_training.ipynb) |
| MediaPipe result display | [12_mediapipe_evaluation.ipynb](notebooks/finetuning_smolvlm256/12_mediapipe_evaluation.ipynb) |

Review server paths, dependencies and adapter initialization before executing a training
notebook. The revised training copies lack saved executed cell outputs; the prediction exports
and executed scoring notebooks provide their result evidence. The configuration JSONs are
provenance records, **not runnable inputs to the existing training CLI**.

From the repository root, score an existing prediction CSV without loading a model or training:

```bash
python scripts/evaluate_256m_saved.py --predictions "/path/to/predictions.csv" --output "/path/to/new_metrics.json"
```

To restrict scoring to the earlier reference paths and matching labels:

```bash
python scripts/evaluate_256m_saved.py --predictions "/path/to/later_predictions.csv" --reference "/path/to/earlier_predictions.csv" --output "/path/to/new_shared_metrics.json"
```

Replace the example paths with local files. The scorer needs Python only, rejects missing or
duplicate paths and mismatched labels, and refuses to overwrite an existing output file.

### Scope and supporting records

Split manifests and image contents were not independently audited in this repository update.
Saved exports do not establish the producing checkpoint for every cached prediction. Recorded
generation and graph latency use different measurement boundaries. No new GPU run, Jetson
benchmark, model export or quantization was performed while preparing the 256M repository files.

- [Experiment narrative](docs/smalvlm256/smolvlm256m_experiments.md)
- [Verified results and source hashes](results/smolvlm_256m/README.md)
- [Configuration records](configs/smolvlm256m/README.md)
- [256M model loading requirements](models/README.md#smolvlm-256m)
- [Model dependency notes](docs/smalvlm256/smolvlm256m_model_loading.md)
- [Added file index](docs/smalvlm256/smolvlm256m_file_index.md)
