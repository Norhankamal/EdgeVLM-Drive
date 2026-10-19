"""Smoke-test both preprocessing versions on a synthetic frame (needs opencv + langgraph)."""
import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("langgraph")

from edgevlm.data.dataset import make_dual_crop
from edgevlm.pipeline import get_preprocessing


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_dual_crop_shapes(version):
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    full, face = make_dual_crop(frame, get_preprocessing(version))
    assert full.shape == (448, 448, 3)
    assert face.shape == (256, 256, 3)


def test_unknown_version():
    with pytest.raises(ValueError):
        get_preprocessing("v9")


def test_only_v2_has_texture_guard():
    assert hasattr(get_preprocessing("v2"), "FACE_CROP_MIN_STD")
    assert not hasattr(get_preprocessing("v1"), "FACE_CROP_MIN_STD")
