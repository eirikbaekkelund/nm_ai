"""
Convert COCO annotations to YOLO format (class-agnostic: all categories -> class 0).
Reads annotations.json, outputs YOLO .txt files alongside images.

Usage: python -m vision_task.detection.convert_coco_to_yolo
"""

import json
import os
import random
import shutil
from pathlib import Path

# Reproducible split
SEED = 42
VAL_FRACTION = 0.15  # ~37 images for validation


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """Convert COCO [x, y, w, h] (pixels) to YOLO [cx, cy, w, h] (normalized 0-1)."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))
    return cx, cy, nw, nh


def main():
    root = Path(__file__).resolve().parents[2]
    coco_json = root / "data" / "coco" / "train" / "annotations.json"
    image_dir = root / "data" / "coco" / "train" / "images"
    out_dir = root / "data" / "shelf_yolo"

    print(f"Loading {coco_json} ...")
    with open(coco_json) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    annotations = coco["annotations"]

    print(
        f"  {len(images)} images, {len(annotations)} annotations, "
        f"{len(coco['categories'])} categories -> collapsing to class 0"
    )

    # Group annotations by image_id
    anns_by_image = {}
    for ann in annotations:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    # Train/val split (by image)
    random.seed(SEED)
    image_ids = sorted(images.keys())
    random.shuffle(image_ids)
    n_val = max(1, int(len(image_ids) * VAL_FRACTION))
    val_ids = set(image_ids[:n_val])
    train_ids = set(image_ids[n_val:])

    print(f"  Split: {len(train_ids)} train, {len(val_ids)} val")

    # Create YOLO directory structure
    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train": 0, "val": 0, "total_boxes": 0, "skipped_empty": 0}

    for img_id in image_ids:
        img_info = images[img_id]
        img_w = img_info["width"]
        img_h = img_info["height"]
        file_name = img_info["file_name"]
        stem = Path(file_name).stem

        split = "val" if img_id in val_ids else "train"
        anns = anns_by_image.get(img_id, [])

        if not anns:
            stats["skipped_empty"] += 1
            continue

        # Copy/symlink image
        src_img = image_dir / file_name
        dst_img = out_dir / "images" / split / file_name
        if not dst_img.exists() and src_img.exists():
            # Use copy on Windows (symlinks need admin), hardlink where possible
            try:
                os.link(str(src_img), str(dst_img))
            except OSError:
                shutil.copy2(str(src_img), str(dst_img))

        # Write YOLO label file
        label_path = out_dir / "labels" / split / f"{stem}.txt"
        with open(label_path, "w") as f:
            for ann in anns:
                cx, cy, nw, nh = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
                # Class 0 for all products (class-agnostic)
                f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                stats["total_boxes"] += 1

        stats[split] += 1

    print(f"\nDone! Output: {out_dir}")
    print(f"  Train images: {stats['train']}")
    print(f"  Val images:   {stats['val']}")
    print(f"  Total boxes:  {stats['total_boxes']}")
    if stats["skipped_empty"]:
        print(f"  Skipped (no annotations): {stats['skipped_empty']}")

    # Verify shelf.yaml path is correct
    yaml_path = Path(__file__).parent / "shelf.yaml"
    print(f"\nYAML config: {yaml_path}")
    print(f"  Verify 'path' resolves to: {out_dir}")


if __name__ == "__main__":
    main()
