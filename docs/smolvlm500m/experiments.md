# SmolVLM 500M experiment progression

All numbers are taken from the saved outputs of the notebooks in
[`notebooks/finetuning_smolvlm500/`](../../notebooks/finetuning_smolvlm500/README.md) and are stored in
[`results/smolvlm_500m/`](../../results/smolvlm_500m/README.md). How the runs differ and which questions are
open: [experiment log](experiment_log.md).

## Starting point

The [256M progression](../smolvlm256m/experiments.md) produced valid labels but missed most drowsy examples
(DROWSY recall 0.00 % and 2.38 % in the single-image graph evaluations), and both selected checkpoints by
validation loss without per-class monitoring. The 500M work therefore changed three things together instead of
only enlarging the model: **dual-crop input** (a full-scene crop and a face crop), **LoRA on both the vision
encoder and the text decoder**, and **checkpoint selection by validation macro-F1** computed from real
generations on a fixed, class-balanced validation subset.

## Initial baseline (notebook 01)

Training used 17,893 rows, validation 3,941 and test 3,579
(subject-disjoint DMD split). LoRA r = 8, alpha = 16 on 128 text-decoder
and 48 vision-encoder attention modules (2,228,224 trainable
parameters, 0.4372 %), full inverse-frequency sampling, a
160-image macro-F1 check and patience 2. The per-epoch validation macro-F1 was
0.6032, 0.6275, 0.6685, 0.5371, 0.6023; epoch 3 was kept. Per-class F1 on the check at epoch 3 was 0.385 for DROWSY and fell
to 0.000 at epoch 4 before partially recovering.

Test result (3,579 images): accuracy 85.14 %, balanced accuracy
74.84 %, macro-F1 74.13 %, unsafe-state miss rate
27.66 %, false-alarm rate 6.37 %, DROWSY recall
0.536. This is the run reported in the interim report.

## On-device validation of the initial baseline

The merged FP16 package (about 968 MB, which needs a 4 GB swap partition on the 4 GB Jetson Nano) was evaluated
on a 120-image balanced subset (30 per class): accuracy 80.00 %, mean latency 4,659 ms, P95 5,623 ms, DROWSY recall
36.67 % (11 of 30; 19 of 30 labelled NORMAL). Source: interim report (Table IX and Sec. VII-D of the nine-page version); see
[`jetson_validation_phase1.json`](../../results/smolvlm_500m/jetson_validation_phase1.json).

## Extended data, larger adapter (notebook 02)

The dataset was extended; training 27,872, validation 5,761, test
5,558 rows (a different, larger test set). LoRA r = 16, alpha = 32
(4,456,448 trainable parameters), weight decay 0.01, a
320-image check and patience 2. Validation macro-F1 was 0.7942, 0.7874, 0.7603, so the
first-epoch adapter was selected and training stopped after three epochs. Test: accuracy 85.44 %,
balanced accuracy 83.0 %, macro-F1 79.56 %, DROWSY recall
0.813, false-alarm rate 10.22 %.

## Code snapshot (notebook 03)

Notebook 03 carries code edits relative to notebook 02 (a larger validation check, patience 3, square-root-damped
sampling, a face-crop texture check) but its saved outputs are identical to notebook 02's, so it has no result of its
own and must not be cited as a separate run.

## Final extended-data run (notebook 04)

The final run adds the current preprocessing (version 2: `extract_face_crop` rejects low-contrast Haar boxes), the
extended instruction that defines every class and states that a visible phone decides PHONE_USAGE regardless of head
pose, square-root-damped inverse-frequency sampling, a 600-image check, patience 3 and
resume support. Training used 27,932 rows. The four recorded validation macro-F1 values were
0.7145, 0.7566, 0.7553, 0.7757, the best being the last. Test: accuracy 86.22 %, balanced accuracy
80.08 %, macro-F1 80.18 %, DROWSY recall
0.809, false-alarm rate 7.75 %, unsafe-state miss
rate 22.45 %.

## Reading the results with care

- The extended-data test set differs from the initial baseline's; compare runs only within one test set.
- Several factors changed together between runs, and the final instruction was written after inspecting the
  test-set errors of the earlier extended-data run, so the final scores may be optimistic.
- The saved output of the final run cannot reconcile its training length (a checkpoint at step 24,440 versus four
  recorded validation evaluations); do not quote epochs or hours for it without the server's trainer state.
- No extended-data model has been benchmarked on the Jetson Nano.
