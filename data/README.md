# Data

No images or CSVs are stored in this repository. The frames come from the **Driver Monitoring
Dataset (DMD)** (Ortega et al., *DMD: A Large-Scale Multi-Modal Driver Monitoring Dataset for
Attention and Alertness Analysis*, ECCV Workshops 2020). They show identifiable drivers and are
subject to the dataset's own access and licence terms, so obtain them from the official source.

## What the training code expects

One CSV per split (`train`, `val`, `test`), paths set in the experiment YAML under `paths:`.

| Column | Meaning |
|---|---|
| `frame_path` | image path, relative to `paths.data_root` (or absolute) |
| `label` | one of `NORMAL`, `DROWSY`, `LOOKING_AWAY`, `PHONE_USAGE` |
| `source_dataset`, `group`, `subject` | provenance columns present in the project's CSVs |

For the training split, augmented images of the DROWSY and LOOKING_AWAY classes are looked up in
`augmented_drowsy_images/` and `augmented_looking_away_images/` under the splits folder.
Validation and test splits contain original frames only.

## How the labels were built

- Face-camera video sampled every 0.5 s; body, hands, mosaic and infrared streams discarded.
- **Subject-level split** (not frame-level) to prevent identity leakage; rare classes balanced across splits.
- **DROWSY:** DMD records raw eye-state intervals. Intervals `closing`, `close`, `opening`
  separated by ≤ 5 frames are merged into one closure episode; an episode lasting ≥ 0.8 s
  (PERCLOS-inspired) is DROWSY. Yawning alone is excluded. Merging raised correctly identified
  DROWSY frames from 470 to 1,190.
- **Distraction → classes:** priority rule `PHONE_USAGE` (phone call / texting) before
  `LOOKING_AWAY` (gaze not on road) before `NORMAL`; frames matching none are excluded.
- Mild augmentation (flip, brightness/contrast, zoom, small rotation) on DROWSY and LOOKING_AWAY,
  training split only.

## Split sizes

| Run | Train | Validation | Test | DROWSY in test |
|---|---:|---:|---:|---:|
| Phase 1 baseline | 17,893 | 3,941 | 3,579 | 168 |
| Phase 2 (extended) | 27,872 – 27,932 | 5,761 | 5,558 | 1,021 |

Phase 2 class counts (final): train 13,992 / 6,287 / 3,088 / 4,565, validation 3,418 / 1,161 / 310 / 872,
test 3,277 / 1,021 / 302 / 958 (NORMAL / DROWSY / LOOKING_AWAY / PHONE_USAGE).


## External sources

**NTHU-DDD (DROWSY).** 18 training subjects, 356 videos; frames every 0.5 s. A frame is DROWSY if it lies
in a run of 1s in `_eye.txt` lasting at least 0.8 s (the same definition as DMD). Only the
`slowBlinkWithNodding` scenario is used (89 videos, 12,203 frames), because the other scenarios are
annotated per acted segment. DROWSY frames are verified with the eye aspect ratio (MediaPipe, threshold
0.24, calibrated on 60 + 60 frames): of 9,920 DROWSY frames, 5,080 are confirmed, 4,393 (44.3 %) become NORMAL and
447 without a face are set aside. Result: 11,756 frames (5,080 DROWSY, 6,676 NORMAL). Subjects are split
70/15/15 at subject level (12/3/3 subjects; 7,957 / 1,820 / 1,979 frames), IDs are prefixed `NTHU_`.

**DrivFace (LOOKING_AWAY).** 606 images of 4 drivers; only head pose is labelled. The 60 left/right
images (27 + 33) are used as LOOKING_AWAY after manual review; the 546 frontal images are excluded.
With 4 subjects no subject-level split is meaningful, so all 60 frames go to the training split.

| Split | DROWSY from NTHU-DDD | NORMAL from NTHU-DDD | DROWSY in extended split |
|---|---:|---:|---:|
| Train | 3,355 | 4,602 | 6,287 (2,932 from DMD) |
| Val | 872 | 948 | 1,161 |
| Test | 853 | 1,126 | 1,021 |

Extended training set: 27,932 rows (NORMAL 13,992 · DROWSY 6,287 · PHONE_USAGE 4,565 · LOOKING_AWAY 3,088).
Validation and test are not augmented: DMD frames plus NTHU-DDD frames from three subjects each.
