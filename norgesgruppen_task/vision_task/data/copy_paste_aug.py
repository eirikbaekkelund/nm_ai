"""
Copy-paste augmentation: remove backgrounds from reference images with rembg,
paste cutouts onto shelf crops to create synthetic training data.

Targets long-tail classes: generates more crops for categories with few shelf examples.

Pipeline:
  1. rembg removes background from each reference image → RGBA cutout
  2. For each target category, paste cutout onto random shelf crop backgrounds
  3. Random transforms: scale (0.5-1.5), rotation (-15°..+15°), position
  4. Save as JPEG crops and append to crops_manifest.csv

Usage:
    python -m vision_task.data.copy_paste_aug
    python -m vision_task.data.copy_paste_aug --crops_per_class 20 --min_shelf_count 30
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from vision_task.config import CROP_SAVE_SIZE

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]


def remove_background(img_bgr):
    """Remove background using rembg, return RGBA numpy array."""
    from rembg import remove

    # rembg expects RGB PIL or bytes; easiest to use bytes
    success, buf = cv2.imencode(".png", img_bgr)
    if not success:
        return None
    result_bytes = remove(buf.tobytes())
    arr = np.frombuffer(result_bytes, dtype=np.uint8)
    rgba = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    return rgba


def paste_cutout_on_background(cutout_rgba, bg_bgr, scale_range=(0.5, 1.5), max_rotation=15):
    """Paste an RGBA cutout onto a BGR background with random transform.

    Returns BGR composite image or None if cutout is too small.
    """
    if cutout_rgba is None or cutout_rgba.shape[2] != 4:
        return None

    bg_h, bg_w = bg_bgr.shape[:2]
    cut_h, cut_w = cutout_rgba.shape[:2]

    # Random scale
    scale = random.uniform(*scale_range)
    new_w = int(cut_w * scale)
    new_h = int(cut_h * scale)

    # Ensure cutout fits in background and isn't tiny
    if new_w < 20 or new_h < 20:
        return None
    new_w = min(new_w, bg_w - 2)
    new_h = min(new_h, bg_h - 2)

    cutout_resized = cv2.resize(cutout_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Random rotation
    angle = random.uniform(-max_rotation, max_rotation)
    if abs(angle) > 1:
        M = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), angle, 1.0)
        cutout_resized = cv2.warpAffine(
            cutout_resized,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

    # Random position
    max_x = bg_w - new_w
    max_y = bg_h - new_h
    if max_x <= 0 or max_y <= 0:
        return None
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    # Alpha composite
    result = bg_bgr.copy()
    alpha = cutout_resized[:, :, 3:4].astype(np.float32) / 255.0
    fg = cutout_resized[:, :, :3].astype(np.float32)
    roi = result[y : y + new_h, x : x + new_w].astype(np.float32)
    blended = fg * alpha + roi * (1 - alpha)
    result[y : y + new_h, x : x + new_w] = blended.astype(np.uint8)

    # Crop the pasted region (with some context) as the training crop
    # Add padding around the paste location
    pad = int(max(new_w, new_h) * 0.1)
    cx1 = max(0, x - pad)
    cy1 = max(0, y - pad)
    cx2 = min(bg_w, x + new_w + pad)
    cy2 = min(bg_h, y + new_h + pad)

    crop = result[cy1:cy2, cx1:cx2]
    return crop


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


def parse_args():
    p = argparse.ArgumentParser(description="Copy-paste augmentation for classifier training")
    p.add_argument("--product_images_dir", type=str, default="data/product_images")
    p.add_argument("--mapping_path", type=str, default="data/category_mapping.json")
    p.add_argument("--coco_json", type=str, default="data/coco/train/annotations.json")
    p.add_argument("--image_dir", type=str, default="data/coco/train/images")
    p.add_argument("--output_dir", type=str, default="data/crops")
    p.add_argument(
        "--crops_per_class",
        type=int,
        default=30,
        help="Target number of synthetic crops per class (only generated for classes below --min_shelf_count)",
    )
    p.add_argument(
        "--min_shelf_count",
        type=int,
        default=50,
        help="Classes with fewer shelf crops than this get synthetic augmentation",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # Load category mapping
    print(f"Loading category mapping from {args.mapping_path}...")
    with open(args.mapping_path, encoding="utf-8") as f:
        mapping = json.load(f)

    # Build product_code -> category_id and collect reference image paths
    code_to_cat = {}
    ref_images_by_cat = {}
    product_images_dir = Path(args.product_images_dir)

    for item in mapping["matched"]:
        if not item["has_images"]:
            continue
        cat_id = item["category_id"]
        code = item["product_code"]
        code_to_cat[code] = cat_id

        product_dir = product_images_dir / code
        if not product_dir.is_dir():
            continue

        imgs = sorted(product_dir.glob("*.jpg"))
        if imgs:
            ref_images_by_cat.setdefault(cat_id, []).extend(imgs)

    print(f"  {len(ref_images_by_cat)} categories with reference images")

    # Count existing shelf crops per class
    print(f"Loading COCO annotations from {args.coco_json}...")
    with open(args.coco_json) as f:
        coco = json.load(f)

    shelf_counts = Counter(ann["category_id"] for ann in coco["annotations"])

    # Find classes that need augmentation
    classes_to_augment = []
    for cat_id in ref_images_by_cat:
        count = shelf_counts.get(cat_id, 0)
        if count < args.min_shelf_count:
            n_needed = args.crops_per_class
            classes_to_augment.append((cat_id, count, n_needed))

    classes_to_augment.sort(key=lambda x: x[1])  # Rarest first
    print(f"  Classes needing augmentation: {len(classes_to_augment)} (below {args.min_shelf_count} shelf crops)")
    if classes_to_augment:
        print(f"  Rarest: cat_id={classes_to_augment[0][0]} with {classes_to_augment[0][1]} shelf crops")

    # Collect background shelf images
    images_info = {img["id"]: img for img in coco["images"]}
    image_dir = Path(args.image_dir)
    bg_paths = [image_dir / img["file_name"] for img in coco["images"]]
    bg_paths = [p for p in bg_paths if p.exists()]
    print(f"  Background images: {len(bg_paths)}")

    # Pre-load a few backgrounds (resized for speed)
    print("Pre-loading background images...")
    backgrounds = []
    for bp in bg_paths[:50]:  # Use up to 50 backgrounds
        bg = cv2.imread(str(bp))
        if bg is not None:
            # Resize to reasonable size for cropping
            bg = resize_max_side(bg, 1024)
            backgrounds.append(bg)
    print(f"  Loaded {len(backgrounds)} backgrounds")

    # Load existing manifest
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "crops_manifest.csv"

    existing_rows = []
    existing_paths = set()
    if manifest_path.exists():
        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows.append(row)
                existing_paths.add(row["path"])
    print(f"  Existing manifest: {len(existing_rows)} entries")

    # Generate synthetic crops
    new_rows = []
    n_generated = 0
    n_rembg_fail = 0
    crop_id = 200000  # High IDs to avoid collisions

    # Cache rembg cutouts per reference image
    cutout_cache = {}

    for cat_id, shelf_count, n_needed in classes_to_augment:
        ref_imgs = ref_images_by_cat.get(cat_id, [])
        if not ref_imgs:
            continue

        generated_for_class = 0
        attempts = 0
        max_attempts = n_needed * 5  # Give up after too many failures

        while generated_for_class < n_needed and attempts < max_attempts:
            attempts += 1

            # Pick random reference image
            ref_path = random.choice(ref_imgs)
            ref_key = str(ref_path)

            # Get or compute cutout
            if ref_key not in cutout_cache:
                ref_bgr = cv2.imread(str(ref_path))
                if ref_bgr is None:
                    cutout_cache[ref_key] = None
                    continue
                cutout_cache[ref_key] = remove_background(ref_bgr)

            cutout = cutout_cache[ref_key]
            if cutout is None:
                n_rembg_fail += 1
                continue

            # Pick random background
            bg = random.choice(backgrounds)

            # Paste with random transform
            crop = paste_cutout_on_background(cutout, bg)
            if crop is None:
                continue

            crop = resize_max_side(crop, CROP_SAVE_SIZE)

            # Save
            crop_name = f"synth_{crop_id}_{cat_id}.jpg"
            if crop_name in existing_paths:
                crop_id += 1
                continue

            crop_path = output_dir / crop_name
            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            new_rows.append(
                {
                    "path": crop_name,
                    "category_id": cat_id,
                    "annotation_id": crop_id,
                    "image_id": 0,  # Synthetic, no real image_id
                    "source": "copy_paste",
                }
            )
            n_generated += 1
            generated_for_class += 1
            crop_id += 1

        if generated_for_class > 0:
            print(
                f"  cat_id={cat_id}: {shelf_count} shelf + {generated_for_class} synthetic = {shelf_count + generated_for_class} total"
            )

    # Write updated manifest
    if new_rows:
        all_rows = existing_rows + new_rows
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "category_id", "annotation_id", "image_id", "source"])
            writer.writeheader()
            writer.writerows(all_rows)

    print(f"\nDone!")
    print(f"  Synthetic crops generated: {n_generated}")
    print(f"  rembg failures: {n_rembg_fail}")
    print(f"  Total manifest entries: {len(existing_rows) + len(new_rows)}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
