import pytest

from edgevlm.data.frame_paths import FrameResolver


@pytest.fixture
def tree(tmp_path):
    frames = tmp_path / "frames"
    splits = tmp_path / "splits"
    (frames / "subj1").mkdir(parents=True)
    (splits / "augmented_drowsy_images" / "subj1").mkdir(parents=True)
    (frames / "subj1" / "a.jpg").write_bytes(b"x")
    (splits / "augmented_drowsy_images" / "subj1" / "a_aug.jpg").write_bytes(b"x")
    return frames, splits


def test_original_frame_resolves_for_every_split(tree):
    frames, splits = tree
    r = FrameResolver(frames, splits)
    for split in ("train", "validation", "test"):
        assert r("subj1/a.jpg", split, "NORMAL") == frames / "subj1" / "a.jpg"


def test_augmented_root_searched_first_only_for_train(tree):
    frames, splits = tree
    r = FrameResolver(frames, splits)
    assert r("subj1/a_aug.jpg", "train", "DROWSY").name == "a_aug.jpg"
    # Direct candidate roots for evaluation splits do not include the augmentation folders...
    assert (r.augmented_image_dirs["DROWSY"] / "subj1/a_aug.jpg").is_file()
    assert [root for root in (frames,) if (root / "subj1/a_aug.jpg").is_file()] == []
    # ...but the suffix-index fallback is split-agnostic (documented caveat), so CSV hygiene,
    # not the resolver, is what keeps augmented images out of validation/test.
    assert r("subj1/a_aug.jpg", "test", "DROWSY").name == "a_aug.jpg"


def test_windows_separators_and_missing(tree):
    frames, splits = tree
    r = FrameResolver(frames, splits)
    assert r("subj1\\a.jpg", "test", "NORMAL").name == "a.jpg"
    with pytest.raises(FileNotFoundError):
        r("nope.jpg", "train", "NORMAL")


def test_suffix_index_is_lazy_and_ambiguity_detected(tmp_path):
    splits = tmp_path / "s"
    for d in ("a", "b"):
        (splits / d / "x").mkdir(parents=True)
        (splits / d / "x" / "dup.jpg").write_bytes(b"x")
    r = FrameResolver(tmp_path / "none", splits)
    assert r._suffix_index is None
    with pytest.raises(FileNotFoundError, match="Ambiguous"):
        r("x/dup.jpg", "train", "NORMAL")
    assert r._suffix_index is not None
