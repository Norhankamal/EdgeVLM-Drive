# SmolVLM 256M results

These are recalculated results from saved prediction CSVs, not new model runs. Each JSON records its exact source filename or archive member and SHA-256. Raw image paths, prediction CSVs and weights are not published.

## Full exported test populations

| Experiment | Images | Accuracy % | Balanced accuracy % | Macro F1 % | DROWSY recall % |
|---|---:|---:|---:|---:|---:|
| `single_image_sampling_direct` | 3,579 | 78.09 | 51.31 | 51.92 | 0.00 |
| `single_image_weighted_direct` | 3,579 | 71.86 | 58.59 | 52.90 | 5.95 |
| `single_image_sampling_graph` | 3,579 | 79.46 | 48.81 | 50.78 | 0.00 |
| `single_image_weighted_graph` | 3,579 | 78.90 | 58.90 | 55.51 | 2.38 |
| `dual_crop_baseline` | 5,558 | 83.68 | 82.80 | 78.16 | 87.07 |
| `revised_no_mediapipe` | 5,558 | 85.16 | 81.89 | 79.63 | 86.88 |
| `mediapipe` | 5,558 | 85.98 | 82.46 | 80.61 | 85.01 |

The first four exports have 3,579 rows; the later three have 5,558. Historical notebook output scores can differ from later CSV exports and must not be silently substituted.

## Matched test paths

| Experiment | Images | Accuracy % | Balanced accuracy % | Macro F1 % | DROWSY recall % |
|---|---:|---:|---:|---:|---:|
| `single_image_sampling_direct` | 3,579 | 78.09 | 51.31 | 51.92 | 0.00 |
| `single_image_weighted_direct` | 3,579 | 71.86 | 58.59 | 52.90 | 5.95 |
| `single_image_sampling_graph` | 3,579 | 79.46 | 48.81 | 50.78 | 0.00 |
| `single_image_weighted_graph` | 3,579 | 78.90 | 58.90 | 55.51 | 2.38 |
| `dual_crop_baseline` | 3,579 | 84.07 | 78.01 | 73.08 | 63.10 |
| `revised_no_mediapipe` | 3,579 | 84.94 | 75.13 | 73.43 | 55.36 |
| `mediapipe` | 3,579 | 86.31 | 75.08 | 74.87 | 50.60 |

All seven exports contain the same 3,579 normalized reference paths with matching labels. The baseline prediction CSV is now included. Matching paths is not an image-content hash check. No split manifests were inspected, and the training settings were not held constant.

## Definitions

- Accuracy: correct predictions / all rows.
- Balanced accuracy: mean recall over true classes present. All four are present in these tables.
- Macro F1: unweighted mean F1 across the four fixed target classes.
- Unsafe-to-safe miss: true DROWSY, LOOKING_AWAY or PHONE_USAGE predicted NORMAL / all unsafe rows.
- Unsafe-class error: any wrong prediction, including INVALID, among unsafe rows.
- False alarm: true NORMAL predicted one of the three unsafe classes / all NORMAL rows. INVALID is separate.
- Confusion matrices have four true-label rows and five prediction columns, including INVALID.
- Stored exact-label validity and parser-failure flags are summarized separately from final labels.
- Mean and P95 retain their source column names. P95 uses linear interpolation. Graph and generation timings differ.

## Recalculate locally

```bash
python scripts/evaluate_256m_saved.py --predictions /private/run/predictions.csv --output /private/run/recomputed.json
python scripts/evaluate_256m_saved.py --predictions /private/later/predictions.csv --reference /private/earlier/langgraph_test_predictions.csv --output /private/later/shared.json
```

The command requires Python only, refuses overwriting an existing output, and does not load weights or train.
