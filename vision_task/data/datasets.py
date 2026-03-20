"""
PreExtractedCropDataset — reads pre-extracted shelf crop JPEGs from disk.
Replaces on-the-fly cropping for training speed.

Usage:
    from vision_task.data.datasets import PreExtractedCropDataset
"""

import csv
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


# Match YOLO conversion script: same seed and split fraction
_SPLIT_SEED = 42
_VAL_FRACTION = 0.15


class PreExtractedCropDataset(Dataset):
    """Reads pre-extracted JPEGs from data/crops/ using crops_manifest.csv."""

    def __init__(self, crops_dir, manifest_path, transform=None, exclude_unknown=True, unknown_category_id=355):
        self.crops_dir = Path(crops_dir)
        self.transform = transform

        # Load manifest
        entries = []
        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cat_id = int(row["category_id"])
                if exclude_unknown and cat_id == unknown_category_id:
                    continue
                entries.append(
                    {
                        "path": row["path"],
                        "category_id": cat_id,
                        "image_id": int(row["image_id"]),
                    }
                )

        self._entries = entries
        self._labels = [e["category_id"] for e in entries]

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, idx):
        entry = self._entries[idx]
        img_path = self.crops_dir / entry["path"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, entry["category_id"]

    @property
    def labels(self):
        """All category_ids, for sampler weight computation."""
        return self._labels

    @property
    def image_ids(self):
        """All image_ids, for image-level splitting."""
        return [e["image_id"] for e in self._entries]


def get_train_val_indices(dataset):
    """Image-level split matching YOLO: SEED=42, VAL_FRACTION=0.15.

    Returns (train_indices, val_indices) into the dataset.
    """
    # Collect unique image IDs
    all_image_ids = sorted(set(dataset.image_ids))

    rng = random.Random(_SPLIT_SEED)
    rng.shuffle(all_image_ids)

    n_val = int(len(all_image_ids) * _VAL_FRACTION)
    val_image_ids = set(all_image_ids[:n_val])

    train_indices = []
    val_indices = []
    for i, img_id in enumerate(dataset.image_ids):
        if img_id in val_image_ids:
            val_indices.append(i)
        else:
            train_indices.append(i)

    return train_indices, val_indices


class TransformWrapper(Dataset):
    """Wraps a dataset subset with a specific transform.

    Accesses the underlying dataset by the provided indices,
    applying its own transform instead of the original.
    """

    def __init__(self, dataset, indices, transform):
        self._dataset = dataset
        self._indices = indices
        self._transform = transform

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        real_idx = self._indices[idx]
        entry = self._dataset._entries[real_idx]
        img_path = self._dataset.crops_dir / entry["path"]
        img = Image.open(img_path).convert("RGB")
        if self._transform:
            img = self._transform(img)
        return img, entry["category_id"]

    @property
    def labels(self):
        """Category IDs for this subset."""
        return [self._dataset._labels[i] for i in self._indices]
