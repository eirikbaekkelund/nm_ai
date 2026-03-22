"""
Convert SKU-110K CSV annotations to YOLO format (single class 0: product).
Expects raw data at data/sku110k/ (manually downloaded).

SKU-110K structure:
  data/sku110k/
    images/
      train_0.jpg, train_1.jpg, ...
      val_0.jpg, ...
      test_0.jpg, ...
    annotations/
      annotations_train.csv
      annotations_val.csv
      annotations_test.csv

CSV columns: image_name, x1, y1, x2, y2, class, image_width, image_height

Usage: python -m vision_task.detection.convert_sku110k
"""

import csv
import os
import shutil
from collections import defaultdict
from pathlib import Path


def sku110k_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """Convert SKU-110K [x1, y1, x2, y2] (pixels) to YOLO [cx, cy, w, h] (normalized)."""
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    return cx, cy, w, h


def convert_split(csv_path, image_dir, out_dir, split, max_images=0, seed=42):
    """Convert one CSV split to YOLO format.

    Args:
        max_images: if > 0, randomly sample this many images from the split.
    """
    import random

    (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Group rows by image
    image_anns = defaultdict(list)
    image_dims = {}

    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 8:
                continue
            img_name = row[0].strip()
            try:
                x1, y1, x2, y2 = float(row[1]), float(row[2]), float(row[3]), float(row[4])
                img_w, img_h = float(row[6]), float(row[7])
            except (ValueError, IndexError):
                continue
            image_anns[img_name].append((x1, y1, x2, y2))
            image_dims[img_name] = (img_w, img_h)

    # Subset selection
    all_image_names = list(image_anns.keys())
    if max_images > 0 and max_images < len(all_image_names):
        rng = random.Random(seed)
        all_image_names = rng.sample(all_image_names, max_images)
        print(f"  Subset: {max_images}/{len(image_anns)} images selected")

    n_images = 0
    n_boxes = 0

    for img_name in all_image_names:
        anns = image_anns[img_name]
        img_w, img_h = image_dims[img_name]
        stem = Path(img_name).stem

        # Link/copy image
        src_img = image_dir / img_name
        dst_img = out_dir / "images" / split / img_name
        if not dst_img.exists() and src_img.exists():
            try:
                os.link(str(src_img), str(dst_img))
            except OSError:
                shutil.copy2(str(src_img), str(dst_img))

        # Write labels
        label_path = out_dir / "labels" / split / f"{stem}.txt"
        with open(label_path, "w") as f:
            for x1, y1, x2, y2 in anns:
                cx, cy, w, h = sku110k_to_yolo(x1, y1, x2, y2, img_w, img_h)
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                n_boxes += 1

        n_images += 1

    return n_images, n_boxes


def main():
    import argparse
    import random

    parser = argparse.ArgumentParser(description="Convert SKU-110K to YOLO format")
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Max images per split (0=all). Saves disk space on small volumes.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subset selection")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    sku_dir = root / "data" / "sku110k"
    out_dir = root / "data" / "sku110k_yolo"

    if not sku_dir.exists():
        print(f"ERROR: SKU-110K data not found at {sku_dir}")
        print("Download from: https://github.com/eg4000/SKU110K_CVPR19")
        print("Place images in data/sku110k/images/ and CSVs in data/sku110k/annotations/")
        return

    ann_dir = sku_dir / "annotations"
    image_dir = sku_dir / "images"

    splits = {
        "train": ann_dir / "annotations_train.csv",
        "val": ann_dir / "annotations_val.csv",
    }

    total_images = 0
    total_boxes = 0

    for split, csv_path in splits.items():
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found, skipping {split}")
            continue

        print(f"Converting {split} split from {csv_path} ...")
        n_images, n_boxes = convert_split(csv_path, image_dir, out_dir, split, args.max_images, args.seed)
        print(f"  {n_images} images, {n_boxes} boxes")
        total_images += n_images
        total_boxes += n_boxes

    print(f"\nDone! Output: {out_dir}")
    print(f"  Total: {total_images} images, {total_boxes} boxes")
    if args.max_images > 0:
        print(f"  (subset: max {args.max_images} images per split)")

    yaml_path = Path(__file__).parent / "sku110k.yaml"
    print(f"\nYAML config: {yaml_path}")
    print(f"  Verify 'path' resolves to: {out_dir}")


if __name__ == "__main__":
    main()
