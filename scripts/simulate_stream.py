#!/usr/bin/env python
"""Replay a recorded sequence through the gate, the scheduled check and the temporal alert layer.

    python scripts/simulate_stream.py --sequence seq.csv --vlm-latency 6.8 --scheduled-interval 10 --out sim.json

``seq.csv`` columns: t (seconds), true, vlm_label, face_found, blurred, and the 26 feature columns.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import pandas as pd

from edgevlm.edge.stream_sim import StreamFrame, simulate_stream
from edgevlm.mesh.features import FEATURE_NAMES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--vlm-latency", type=float, required=True, help="seconds per VLM call on the target device")
    ap.add_argument("--mesh-latency", type=float, default=0.115)
    ap.add_argument("--scheduled-interval", type=float, default=10.0)
    ap.add_argument("--no-triage", action="store_true", help="use the hard rules only")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.sequence)
    frames = []
    for _, r in df.iterrows():
        found = bool(r.get("face_found", 1))
        feats = {k: float(r[k]) for k in FEATURE_NAMES} if found else None
        frames.append(StreamFrame(t=float(r["t"]), true_label=r["true"], features=feats,
                                  vlm_label=r["vlm_label"], blurred=bool(r.get("blurred", 0))))
    res = simulate_stream(frames, args.vlm_latency, args.scheduled_interval, use_triage=not args.no_triage,
                          mesh_latency_s=args.mesh_latency)
    payload = {**{k: v for k, v in asdict(res).items() if k not in ("alerts", "episodes")},
               "skip_rate": res.skip_rate, "alerts": [asdict(a) for a in res.alerts], "episodes": res.episodes}
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"frames={res.frames} processed={res.processed} vlm_calls={res.vlm_calls} skipped={res.skipped} "
          f"alerts={len(res.alerts)}")


if __name__ == "__main__":
    main()
