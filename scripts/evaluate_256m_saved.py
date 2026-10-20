"""Recompute 256M saved-prediction metrics; no training or model loading."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edgevlm.evaluation.saved_256m import file_sha256, read_predictions, score, shared_subset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Optional CSV defining matched paths and labels")
    parser.add_argument("--output", type=Path, required=True, help="New JSON file; existing files are refused")
    args = parser.parse_args()
    records = read_predictions(args.predictions)
    source = {"file": args.predictions.name, "sha256": file_sha256(args.predictions)}
    if args.reference:
        records = shared_subset(records, read_predictions(args.reference))
        source["reference"] = {"file": args.reference.name, "sha256": file_sha256(args.reference)}
    result = {"source": source, "metrics": score(records)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Scored {len(records)} saved rows; accuracy {result['metrics']['accuracy_percent']:.2f}%")


if __name__ == "__main__":
    main()
