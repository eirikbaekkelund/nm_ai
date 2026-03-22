"""
Pre-extract all shelf crops to disk as small JPEGs.
Reads each shelf image once, crops all annotations with padding, saves to data/crops/.
Generates crops_manifest.csv for the CropDataset.

Usage: python -m vision_task.data.preprocess_crops
"""

import csv
import json
import sys
from pathlib import Path

import cv2

from vision_task.config import CROP_BUFFER, CROP_SAVE_SIZE

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def resize_max_side(img, max_side):
    """Resize so the longest side equals max_side, preserving aspect ratio."""
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    if w >= h:
        new_w = max_side
        new_h = int(h * max_side / w)
    else:
        new_h = max_side
        new_w = int(w * max_side / h)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def main():
    root = Path(__file__).resolve().parents[2]
    coco_json = root / "data" / "coco" / "train" / "annotations.json"
    image_dir = root / "data" / "coco" / "train" / "images"
    out_dir = root / "data" / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {coco_json} ...")
    with open(coco_json) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    annotations = coco["annotations"]

    # Group annotations by image_id to load each image only once
    anns_by_image = {}
    for ann in annotations:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    print(f"  {len(images)} images, {len(annotations)} annotations")
    print(f"  Crop buffer: {CROP_BUFFER}, max side: {CROP_SAVE_SIZE}")

    manifest_path = out_dir / "crops_manifest.csv"
    manifest_rows = []

    n_crops = 0
    n_errors = 0

    for img_idx, (img_id, img_info) in enumerate(sorted(images.items())):
        file_name = img_info["file_name"]
        img_path = image_dir / file_name

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARNING: Could not read {img_path}")
            continue

        img_h, img_w = img.shape[:2]
        anns = anns_by_image.get(img_id, [])

        for ann in anns:
            ann_id = ann["id"]
            cat_id = ann["category_id"]
            x, y, w, h = ann["bbox"]

            # Add buffer padding
            pad_w = int(w * CROP_BUFFER)
            pad_h = int(h * CROP_BUFFER)
            x1 = max(0, int(x) - pad_w)
            y1 = max(0, int(y) - pad_h)
            x2 = min(img_w, int(x + w) + pad_w)
            y2 = min(img_h, int(y + h) + pad_h)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                n_errors += 1
                continue

            # Resize to max side
            crop = resize_max_side(crop, CROP_SAVE_SIZE)

            # Save
            crop_name = f"{ann_id}_{cat_id}.jpg"
            crop_path = out_dir / crop_name
            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            manifest_rows.append(
                {
                    "path": crop_name,
                    "category_id": cat_id,
                    "annotation_id": ann_id,
                    "image_id": img_id,
                    "source": "shelf",
                }
            )
            n_crops += 1

        if (img_idx + 1) % 50 == 0:
            print(f"  Processed {img_idx + 1}/{len(images)} images, {n_crops} crops so far")

    # Write manifest
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "category_id", "annotation_id", "image_id", "source"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDone! Output: {out_dir}")
    print(f"  Crops extracted: {n_crops}")
    if n_errors:
        print(f"  Skipped (empty crop): {n_errors}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
