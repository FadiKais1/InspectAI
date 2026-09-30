"""Framework-independent MVTec AD dataset (NumPy / Pillow)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from inspectai.data.transforms import PairedTransform
from inspectai.data.types import (
    MVTecError,
    MVTecLayoutError,
    MVTecPairingError,
    MVTecSample,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VALID_SPLITS = ("train", "test")


@dataclass(frozen=True)
class _IndexEntry:
    image_path: Path
    mask_path: Path | None
    label: int
    defect_type: str


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def _list_class_dirs(split_dir: Path) -> list[Path]:
    return sorted(path for path in split_dir.iterdir() if path.is_dir())


def _list_images(class_dir: Path) -> list[Path]:
    return sorted(path for path in class_dir.iterdir() if _is_image_file(path))


def _mask_key(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_mask"):
        return stem[: -len("_mask")]
    return stem


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        array = np.asarray(img.convert("RGB"))
    if array.ndim != 3 or array.shape[2] != 3:
        raise MVTecError(f"Expected HWC RGB image, got shape {array.shape} at {path}")
    return array


def _to_binary_mask(array: np.ndarray, source: Path) -> np.ndarray:
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise MVTecPairingError(f"Mask must be 2D after squeeze, got {array.shape} at {source}")
    return (array > 0).astype(np.uint8)


def _load_binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        array = np.asarray(img)
    return _to_binary_mask(array, path)


def _zero_mask(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), dtype=np.uint8)


class MVTecDataset:
    """Indexed MVTec AD split with lazy native-resolution loading.

    ``root`` is the directory that contains category folders (for example
    ``data/raw``). Defect folder names are discovered from disk and are not
    hard-coded.
    """

    def __init__(
        self,
        root: Path,
        category: str,
        split: Literal["train", "test"],
        transform: PairedTransform | None = None,
    ) -> None:
        self.root = Path(root)
        self.category = category
        self.split = split
        self.transform = transform
        self.category_dir = self.root / category

        if split not in VALID_SPLITS:
            raise MVTecLayoutError(
                f"Invalid split {split!r}; expected one of {VALID_SPLITS}."
            )
        if not self.category_dir.is_dir():
            raise MVTecLayoutError(f"Missing category directory: {self.category_dir}")

        self._entries: tuple[_IndexEntry, ...]
        self._defect_types: tuple[str, ...]
        if split == "train":
            self._entries, self._defect_types = self._index_train()
        else:
            self._entries, self._defect_types = self._index_test()

    @property
    def defect_types(self) -> tuple[str, ...]:
        """Defect class names discovered for this split (empty on train)."""
        return self._defect_types

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> MVTecSample:
        entry = self._entries[index]
        image = _load_rgb(entry.image_path)
        height, width = image.shape[:2]

        if entry.mask_path is None:
            mask = _zero_mask(height, width)
        else:
            mask = _load_binary_mask(entry.mask_path)
            if mask.shape != (height, width):
                raise MVTecPairingError(
                    "Image/mask spatial size mismatch: "
                    f"image {height}x{width} at {entry.image_path} vs "
                    f"mask {mask.shape[0]}x{mask.shape[1]} at {entry.mask_path}"
                )

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return MVTecSample(
            image=image,
            mask=mask,
            label=entry.label,
            defect_type=entry.defect_type,
            image_path=entry.image_path,
        )

    def _index_train(self) -> tuple[tuple[_IndexEntry, ...], tuple[str, ...]]:
        train_dir = self.category_dir / "train"
        if not train_dir.is_dir():
            raise MVTecLayoutError(f"Missing train directory: {train_dir}")

        class_dirs = _list_class_dirs(train_dir)
        class_names = [path.name for path in class_dirs]
        unexpected = [name for name in class_names if name != "good"]
        if unexpected:
            raise MVTecLayoutError(
                "Unexpected train class folders (MVTec train must be good-only): "
                f"{unexpected} under {train_dir}"
            )

        good_dir = train_dir / "good"
        if not good_dir.is_dir():
            raise MVTecLayoutError(f"Missing train/good directory: {good_dir}")

        images = _list_images(good_dir)
        if not images:
            raise MVTecLayoutError(f"No training images found in {good_dir}")

        entries = tuple(
            _IndexEntry(
                image_path=path.resolve(),
                mask_path=None,
                label=0,
                defect_type="good",
            )
            for path in images
        )
        self._assert_unique_paths(entries)
        return entries, ()

    def _index_test(self) -> tuple[tuple[_IndexEntry, ...], tuple[str, ...]]:
        test_dir = self.category_dir / "test"
        if not test_dir.is_dir():
            raise MVTecLayoutError(f"Missing test directory: {test_dir}")

        good_dir = test_dir / "good"
        if not good_dir.is_dir():
            raise MVTecLayoutError(f"Missing test/good directory: {good_dir}")

        class_dirs = _list_class_dirs(test_dir)
        defect_dirs = [path for path in class_dirs if path.name != "good"]
        defect_types = tuple(path.name for path in defect_dirs)
        gt_dir = self.category_dir / "ground_truth"

        if defect_dirs and not gt_dir.is_dir():
            raise MVTecLayoutError(
                f"Missing ground_truth directory required for defective test images: {gt_dir}"
            )

        entries: list[_IndexEntry] = []
        for class_dir in class_dirs:
            if class_dir.name == "good":
                for image_path in _list_images(class_dir):
                    entries.append(
                        _IndexEntry(
                            image_path=image_path.resolve(),
                            mask_path=None,
                            label=0,
                            defect_type="good",
                        )
                    )
                continue
            entries.extend(self._pair_defect_class(class_dir, gt_dir / class_dir.name))

        if gt_dir.is_dir():
            self._assert_no_unmatched_gt_classes(gt_dir, set(defect_types))

        index = tuple(entries)
        self._assert_unique_paths(index)
        return index, defect_types

    def _pair_defect_class(
        self, test_class_dir: Path, gt_class_dir: Path
    ) -> list[_IndexEntry]:
        images = _list_images(test_class_dir)
        if not gt_class_dir.is_dir():
            raise MVTecPairingError(
                f"Missing ground-truth folder for defect {test_class_dir.name!r}: {gt_class_dir}"
            )

        masks = _list_images(gt_class_dir)
        invalid_names = [path.name for path in masks if not path.stem.endswith("_mask")]
        if invalid_names:
            raise MVTecPairingError(
                f"Ground-truth files must use a '_mask' stem suffix under {gt_class_dir}: "
                f"{invalid_names}"
            )

        image_stems = {path.stem: path for path in images}
        mask_stems = {_mask_key(path): path for path in masks}
        if len(mask_stems) != len(masks):
            raise MVTecPairingError(f"Duplicate mask stems under {gt_class_dir}")

        missing = sorted(set(image_stems) - set(mask_stems))
        extra = sorted(set(mask_stems) - set(image_stems))
        if missing:
            raise MVTecPairingError(
                f"Missing masks for {test_class_dir.name!r}: {missing} "
                f"(looked in {gt_class_dir})"
            )
        if extra:
            raise MVTecPairingError(
                f"Unmatched masks for {test_class_dir.name!r}: {extra} under {gt_class_dir}"
            )

        return [
            _IndexEntry(
                image_path=image_path.resolve(),
                mask_path=mask_stems[stem].resolve(),
                label=1,
                defect_type=test_class_dir.name,
            )
            for stem, image_path in sorted(image_stems.items())
        ]

    def _assert_no_unmatched_gt_classes(
        self, gt_dir: Path, defect_types: set[str]
    ) -> None:
        extra_classes = [
            path.name for path in _list_class_dirs(gt_dir) if path.name not in defect_types
        ]
        if extra_classes:
            raise MVTecPairingError(
                f"Unmatched ground_truth class folders {extra_classes} under {gt_dir}"
            )

    @staticmethod
    def _assert_unique_paths(entries: Sequence[_IndexEntry]) -> None:
        seen: dict[Path, int] = {}
        duplicates: list[Path] = []
        for entry in entries:
            key = entry.image_path
            if key in seen:
                duplicates.append(key)
            seen[key] = seen.get(key, 0) + 1
        if duplicates:
            raise MVTecLayoutError(f"Duplicate indexed image paths: {duplicates}")
