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

## How the labels were built (summary of the interim report, Sec. VI-C)

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
