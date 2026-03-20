"""
ProductReferenceDataset — loads multi-angle reference photos per product,
mapped to category_id via category_mapping.json.

Usage:
    from vision_task.data.reference_dataset import ProductReferenceDataset
"""

import json
from pathlib import Path
from PIL import Image
from typing import List
from torch.utils.data import Dataset


class ProductReferenceDataset(Dataset):
    """Loads reference product images with category_id labels.

    Uses category_mapping.json to bridge product_code → category_id.
    Each product may have multiple angle images (main, front, back, etc.).
    CUSTOM_xxx dirs are excluded (no category mapping).
    """

    def __init__(self, product_images_dir: str, mapping_path: str, transform=None):
        self.product_images_dir = Path(product_images_dir)
        self.transform = transform

        with open(mapping_path, encoding="utf-8") as f:
            mapping = json.load(f)

        # Build (image_path, category_id) pairs for all matched products with images
        entries = []
        for item in mapping["matched"]:
            if not item["has_images"]:
                continue

            product_dir: Path = self.product_images_dir / item["product_code"]
            if not product_dir.is_dir():
                continue

            cat_id = item["category_id"]
            for img_file in sorted(product_dir.glob("*.jpg")):
                entries.append(
                    {
                        "path": img_file,
                        "category_id": cat_id,
                        "product_code": item["product_code"],
                    }
                )

        self._entries = entries
        self._labels = [e["category_id"] for e in entries]

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, idx):
        entry = self._entries[idx]
        img = Image.open(entry["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, entry["category_id"]

    @property
    def labels(self) -> List[int]:
        """All category_ids, for sampler weight computation."""
        return self._labels
