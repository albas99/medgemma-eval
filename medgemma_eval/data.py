"""Dataset loading compatible with the LO-VLM data layout.

The LO-VLM dataset (`lo-vlm/dataset_full/hf_dataset/`) is a `save_to_disk`
Arrow dataset whose columns are strings, NOT the "traditional" HF format
with an embedded `Image` feature:

    split:     string
    img_path:  string   (relative filename, e.g. "sample_0001.png")
    label:     string   (e.g. "AMD_late,ERM")
    diagnoses: List[string]
    desc:      string   (the ground-truth report / layer information)

Images live in a separate folder and are loaded manually, exactly like
LO-VLM's `inference.py` does.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from datasets import load_dataset, load_from_disk
from PIL import Image


@dataclass
class DatasetSpec:
    """Column names, mirroring LO-VLM defaults."""

    image_column: str = "img_path"
    label_column: str = "label"
    report_column: str = "desc"
    images_dir: str = ""  # directory the img_path values are relative to


def load_dataset_any(path_or_name: str, split: Optional[str] = None):
    """Load a dataset from disk (LO-VLM style) or from the HF hub.

    - If `path_or_name` is an existing directory, `load_from_disk` is used.
      If it contains train/val/test subdirectories (e.g. hf_dataset/), the
      requested `split` is loaded from that subdirectory.
    - Otherwise it is treated as an HF hub dataset name.
    """
    path = Path(path_or_name)
    if path.exists() and path.is_dir():
        for split_name in (split,) if split else ():
            split_dir = path / split_name
            if split_name and split_dir.is_dir():
                return load_from_disk(str(split_dir))
        return load_from_disk(str(path))
    return load_dataset(path_or_name, split=split if split else "train")


def load_image(example: dict, spec: DatasetSpec) -> Image.Image:
    """Load one image, following the LO-VLM convention (img_path + imgs_dir)."""
    value = example[spec.image_column]

    # Defensive: some HF exports store dicts like {"path": ..., "bytes": ...}
    if isinstance(value, dict):
        if value.get("bytes"):
            import io

            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        value = value.get("path") or value.get("url") or ""

    if isinstance(value, Image.Image):
        return value.convert("RGB")

    path = str(value)
    if spec.images_dir and not os.path.isabs(path):
        path = os.path.join(spec.images_dir, path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Image not found: {path}. Pass --images_dir pointing to the folder "
            f"containing '{example.get(spec.image_column)}'."
        )
    return Image.open(path).convert("RGB")


def check_columns(dataset: Any, spec: DatasetSpec) -> None:
    cols = list(dataset.column_names)
    missing = [
        name
        for name in (spec.image_column, spec.report_column)
        if name not in cols
    ]
    if missing:
        raise ValueError(
            f"Dataset columns {cols} are missing {missing}. "
            f"Set --image-column / --report-column accordingly."
        )
