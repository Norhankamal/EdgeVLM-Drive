"""Inference / preprocessing pipeline (LangGraph state graph + dual-crop preprocessing).

Two versions are kept side by side because the fine-tuning experiments were run on
different preprocessing code:

``"v1"``  ``legacy/langgraph_pipeline_v1.py`` - used by the Phase 1 baseline and Phase 2 v1.
``"v2"``  ``langgraph_pipeline.py``           - current; ``extract_face_crop`` additionally
          rejects Haar boxes that sit on flat, textureless surfaces (``FACE_CROP_MIN_STD``)
          and the temporal decision logic uses per-class mechanisms.

Both modules expose the same preprocessing functions (``enhance_frame``,
``resize_full_frame``, ``extract_face_crop``), so dataset and evaluation code can take
either one.
"""

from __future__ import annotations

import importlib
from types import ModuleType

_MODULES = {
    "v1": "edgevlm.pipeline.legacy.langgraph_pipeline_v1",
    "v2": "edgevlm.pipeline.langgraph_pipeline",
}


def get_preprocessing(version: str = "v2") -> ModuleType:
    """Return the pipeline module for ``version`` (imported lazily; needs ``langgraph``)."""
    try:
        return importlib.import_module(_MODULES[version])
    except KeyError:
        raise ValueError(f"Unknown preprocessing version {version!r}; expected one of {sorted(_MODULES)}") from None
