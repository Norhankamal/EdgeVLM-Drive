#!/usr/bin/env python
"""Merge the best adapter into the base model and export a standalone FP16 PyTorch package.

    python scripts/export_model.py --adapter runs/<name>/best_by_macro_f1 --out export/smolvlm_500m [--zip]
"""
from __future__ import annotations

import argparse

from edgevlm.constants import IMAGE_SIZE, MODEL_ID
from edgevlm.deployment.export import merge_and_export


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--model-cache", default=None)
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()
    merge_and_export(a.adapter, a.out, a.model_id, a.model_cache, a.image_size, a.zip)


if __name__ == "__main__":
    main()
