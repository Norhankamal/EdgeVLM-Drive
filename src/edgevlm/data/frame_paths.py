"""Resolve the ``frame_path`` column of the split CSVs to real image files on disk.

Ported from the notebooks' ``resolve_frame_path`` (Section 2). The CSVs store paths that
were written on different machines and, for the augmented classes, point into separate
augmentation folders, so a path is resolved by trying, in order:

1. the path itself, if it is absolute and exists;
2. candidate roots - for the *train* split the label-specific augmentation folder
   (DROWSY / LOOKING_AWAY), then the splits folder, then the frame root; for
   validation/test only the frame root;
3. a suffix index over the splits folder (matches on the last up-to-6 path components).

Caveat (inherited from the notebooks): step 3 applies to *every* split, so the resolver does
not by itself stop a validation/test CSV row from resolving to an augmented file. Keeping
augmented images out of evaluation is a property of the CSVs (augmentation was applied to the
training split only), not something this class enforces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
DEFAULT_AUGMENTED_SUBDIRS = {
    "DROWSY": "augmented_drowsy_images",
    "LOOKING_AWAY": "augmented_looking_away_images",
}
_MAX_SUFFIX_PARTS = 6


class FrameResolver:
    def __init__(
        self,
        frame_root: Path | str,
        splits_dir: Path | str,
        augmented_dir: Optional[Path | str] = None,
        augmented_subdirs: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.frame_root = Path(frame_root)
        self.splits_dir = Path(splits_dir)
        self.augmented_dir = Path(augmented_dir) if augmented_dir is not None else self.splits_dir
        subdirs = DEFAULT_AUGMENTED_SUBDIRS if augmented_subdirs is None else augmented_subdirs
        self.augmented_image_dirs = {label: self.augmented_dir / sub for label, sub in subdirs.items()}
        self._suffix_index: Optional[dict[str, list[Path]]] = None

    # -- suffix index (built once, on first fallback) -------------------------------------
    def _build_suffix_index(self) -> dict[str, list[Path]]:
        index: dict[str, list[Path]] = {}
        if not self.augmented_dir.exists():
            return index
        for image_path in self.augmented_dir.rglob("*"):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            parts = image_path.relative_to(self.augmented_dir).as_posix().split("/")
            for start in range(max(0, len(parts) - _MAX_SUFFIX_PARTS), len(parts)):
                index.setdefault("/".join(parts[start:]), []).append(image_path)
        return index

    @property
    def suffix_index(self) -> dict[str, list[Path]]:
        if self._suffix_index is None:
            self._suffix_index = self._build_suffix_index()
        return self._suffix_index

    # -- resolution -----------------------------------------------------------------------
    def resolve(self, frame_path, split_name: str, label: Optional[str] = None) -> Path:
        path_text = str(frame_path).strip().replace("\\", "/")
        relative_path = Path(path_text)

        if relative_path.is_absolute() and relative_path.is_file():
            return relative_path

        preferred_dir = self.augmented_image_dirs.get(str(label).strip().upper())

        if split_name == "train":
            roots = []
            if preferred_dir is not None:
                roots.append(preferred_dir)
            roots.extend([self.augmented_dir, self.frame_root])
        else:
            roots = [self.frame_root]

        for root in roots:
            candidate = root / relative_path
            if candidate.is_file():
                return candidate

        key = relative_path.as_posix().lstrip("./")
        matches = list(self.suffix_index.get(key, []))

        if preferred_dir is not None:
            preferred = [item for item in matches if preferred_dir in item.parents]
            if preferred:
                return preferred[0]

        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise FileNotFoundError(f"Ambiguous path {frame_path} for label {label}: {matches[:5]}")
        raise FileNotFoundError(f"Could not resolve {frame_path} for split {split_name}")

    __call__ = resolve
