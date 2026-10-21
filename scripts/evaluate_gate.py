#!/usr/bin/env python
"""Evaluate the face-mesh gate offline on saved single-image predictions.

    python scripts/evaluate_gate.py --predictions preds.csv --features mesh_features.csv \
        --model-ms 6807 --mesh-ms 115 --out gate_results.json

``preds.csv``    : frame_path, true, pred
``mesh_features``: frame_path, face_found (0/1), blurred (0/1) and the 26 feature columns
                   (see ``edgevlm.mesh.features.FEATURE_NAMES``); rows without a face may leave features empty.

Reports the model alone, the hard-rule gate and the hard-rule + any-doubt-triage gate.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from edgevlm.edge.gate_eval import evaluate_gate, gate_calls
from edgevlm.mesh.features import FEATURE_NAMES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--model-ms", type=float, required=True, help="mean model time per frame on the target device")
    ap.add_argument("--mesh-ms", type=float, default=115.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    preds = pd.read_csv(args.predictions)
    feats = pd.read_csv(args.features)
    df = preds.merge(feats, on="frame_path", how="inner", validate="one_to_one")
    if len(df) != len(preds):
        raise SystemExit(f"{len(preds) - len(df)} predictions have no mesh-feature row")

    rows = []
    for _, r in df.iterrows():
        found = bool(r.get("face_found", 1))
        rows.append({k: float(r[k]) for k in FEATURE_NAMES} if found else None)
    blurred = df["blurred"].astype(bool).tolist() if "blurred" in df else [False] * len(df)

    out = {}
    all_calls = [True] * len(df)
    out["model_only"] = evaluate_gate(df["true"], df["pred"], all_calls, args.model_ms, 0.0)
    for name, use_triage in (("mesh_hard_rules", False), ("mesh_rules_plus_triage", True)):
        calls = gate_calls(rows, blurred, use_triage=use_triage)
        out[name] = evaluate_gate(df["true"], df["pred"], calls, args.model_ms, args.mesh_ms)

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    for name, r in out.items():
        print(f"{name:26s} acc={100 * r['accuracy']:.2f}% skipped={100 * r['skip_rate']:.1f}% "
              f"unsafe_skipped={r['unsafe_frames_skipped']} ms/frame={r['estimated_ms_per_frame']:.0f}")


if __name__ == "__main__":
    main()
