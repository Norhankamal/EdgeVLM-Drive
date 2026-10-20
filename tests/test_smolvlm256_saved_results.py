"""CPU-only regression tests for the 256M contribution; also runnable with unittest."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edgevlm.evaluation.saved_256m import (
    LABELS,
    metrics_from_matrix,
    normalize_path,
    read_predictions,
    score,
    shared_subset,
)


class SavedPredictionTests(unittest.TestCase):
    def csv_file(self, folder, rows, name="predictions.csv"):
        path = Path(folder) / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["frame_path", "true_label", "prediction"])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_perfect_predictions(self):
        rows = {str(i): {"true": label, "pred": label} for i, label in enumerate(LABELS)}
        metrics = score(rows)
        self.assertEqual(metrics["accuracy_percent"], 100)
        self.assertEqual(metrics["macro_f1_percent"], 100)
        self.assertEqual(metrics["unsafe_to_safe_miss_count"], 0)

    def test_invalid_and_unsafe_confusion_are_distinct(self):
        rows = {
            "a": {"true": "NORMAL", "pred": "INVALID"},
            "b": {"true": "DROWSY", "pred": "NORMAL"},
            "c": {"true": "PHONE_USAGE", "pred": "LOOKING_AWAY"},
        }
        metrics = score(rows)
        self.assertEqual(metrics["invalid_prediction_count"], 1)
        self.assertEqual(metrics["unsafe_to_safe_miss_rate_percent"], 50)
        self.assertEqual(metrics["unsafe_class_error_rate_percent"], 100)
        self.assertEqual(metrics["false_alarm_count"], 0)

    def test_no_silent_identity_matching(self):
        self.assertEqual(normalize_path("/home/jovyan/a/f.jpg"), "a/f.jpg")
        self.assertNotEqual(normalize_path("a/f.jpg"), normalize_path("b/f.jpg"))
        self.assertNotEqual(normalize_path("A/f.jpg"), normalize_path("a/f.jpg"))

    def test_duplicate_paths_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.csv_file(folder, [
                {"frame_path": "a", "true_label": "NORMAL", "prediction": "NORMAL"},
                {"frame_path": "/home/jovyan/a", "true_label": "NORMAL", "prediction": "NORMAL"},
            ])
            with self.assertRaises(ValueError):
                read_predictions(path)

    def test_empty_csv_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                read_predictions(self.csv_file(folder, []))

    def test_unknown_label_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.csv_file(folder, [{"frame_path": "a", "true_label": "NORMAL", "prediction": "OTHER"}])
            with self.assertRaises(ValueError):
                read_predictions(path)

    def test_shared_missing_or_mismatched_labels_rejected(self):
        reference = {"a": {"true": "NORMAL", "pred": "NORMAL"}}
        with self.assertRaises(ValueError):
            shared_subset({}, reference)
        with self.assertRaises(ValueError):
            shared_subset({"a": {"true": "DROWSY", "pred": "NORMAL"}}, reference)
        self.assertEqual(shared_subset(reference, reference), reference)

    def test_bad_matrices_rejected(self):
        for matrix in ([[0] * 5 for _ in range(4)], [[1, 2]], [[-1] * 5 for _ in range(4)]):
            with self.assertRaises(ValueError):
                metrics_from_matrix(matrix)

    def test_p95_and_explicit_flags(self):
        rows = {str(i): {"true": "NORMAL", "pred": "NORMAL", "latency_ms": i,
                         "exact_label_valid": True, "parser_failure": False} for i in range(11)}
        metrics = score(rows)
        self.assertEqual(metrics["recorded_latency"]["latency_ms"]["p95_ms"], 9.5)
        self.assertEqual(metrics["exact_label_valid_percent"], 100)
        self.assertEqual(metrics["parser_failure_percent"], 0)

    def test_published_counts_independently_reconcile(self):
        result_dir = ROOT / "results" / "smolvlm_256m"
        excluded = ("source_manifest.json", "shared_test_comparison.json")
        files = [p for p in result_dir.glob("*.json") if p.name not in excluded]
        self.assertEqual(len(files), 7)
        for path in files:
            metrics = json.loads(path.read_text(encoding="utf-8"))["metrics"]
            matrix = metrics["confusion_matrix"]["matrix"]
            n = sum(map(sum, matrix))
            correct = sum(matrix[i][i] for i in range(4))
            self.assertEqual(metrics["images"], n)
            self.assertAlmostEqual(metrics["accuracy_percent"], 100 * correct / n)
            f1s = []
            for i, label in enumerate(LABELS):
                support, predicted = sum(matrix[i]), sum(row[i] for row in matrix)
                p = matrix[i][i] / predicted if predicted else 0
                r = matrix[i][i] / support if support else 0
                f1s.append(2 * p * r / (p + r) if p + r else 0)
                self.assertEqual(metrics["per_class"][label]["support"], support)
            self.assertAlmostEqual(metrics["macro_f1_percent"], sum(f1s) * 25)

    def test_known_full_results(self):
        expected = {"dual_crop_baseline": 83.68, "revised_no_mediapipe": 85.16, "mediapipe": 85.98}
        for key, accuracy in expected.items():
            result = json.loads((ROOT / "results" / "smolvlm_256m" / f"{key}.json").read_text())
            self.assertEqual(result["metrics"]["images"], 5558)
            self.assertEqual(round(result["metrics"]["accuracy_percent"], 2), accuracy)

    def test_shared_comparison(self):
        result = json.loads((ROOT / "results/smolvlm_256m/shared_test_comparison.json").read_text())
        self.assertEqual(len(result["experiments"]), 7)
        for value in result["experiments"].values():
            self.assertEqual(value["metrics"]["images"], 3579)
            self.assertEqual(value["label_mismatches"], 0)
            supports = [p["support"] for p in value["metrics"]["per_class"].values()]
            self.assertEqual(supports, [2151, 168, 302, 958])

    def test_configuration_records_are_not_cli_configs(self):
        paths = list((ROOT / "configs/smolvlm256m").glob("*.json"))
        self.assertEqual(len(paths), 12)
        for path in paths:
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(record["runnable_by_scripts_train_py"])
            self.assertTrue((ROOT / record["notebook"]).is_file())

    def test_notebook_execution_counts_preserved(self):
        manifest = json.loads((ROOT / "notebooks/finetuning_smolvlm256/source_manifest.json").read_text())
        for record in manifest["notebooks"]:
            nb = json.loads((ROOT / record["repository_file"]).read_text(encoding="utf-8"))
            executed = sum(c.get("execution_count") is not None for c in nb["cells"])
            self.assertEqual(record["executed_code_cells"], executed)
            self.assertNotIn("widgets", nb["metadata"])
            for cell in nb["cells"]:
                for output in cell.get("outputs", []):
                    self.assertTrue(set(output.get("data", {})).issubset({"text/plain"}))

    def test_cli_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.csv_file(folder, [{"frame_path": "a", "true_label": "NORMAL", "prediction": "NORMAL"}])
            out = Path(folder) / "result.json"
            cmd = [sys.executable, str(ROOT / "scripts/evaluate_256m_saved.py"),
                   "--predictions", str(path), "--output", str(out)]
            first = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            second = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)


if __name__ == "__main__":
    unittest.main()
