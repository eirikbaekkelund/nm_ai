"""
Mine additional training crops by running YOLO on shelf images and matching to GT.

YOLO bboxes differ slightly from GT bboxes (tighter/looser, different aspect ratios).
Training the classifier on YOLO-perspective crops bridges the train/inference gap.

For each YOLO prediction with IoU > threshold against a GT box, we:
  1. Crop from the original image using the YOLO bbox (with 5% padding)
  2. Assign the GT label (category_id) from the matched GT box
  3. Save to data/crops/ and append to crops_manifest.csv

Usage:
    python -m vision_task.data.mine_yolo_crops
    python -m vision_task.data.mine_yolo_crops --yolo_weights models/yolo_best.pt --iou_thresh 0.5
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

from vision_task.config import CROP_BUFFER, CROP_SAVE_SIZE, DETECTOR_IMGSZ

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]


def compute_iou(box_a, box_b):
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def coco_to_xyxy(bbox):
    """Convert COCO [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


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
    p = argparse.ArgumentParser(description="Mine YOLO crops for classifier training")
    p.add_argument("--yolo_weights", type=str, default="models/yolo_best.pt")
    p.add_argument("--coco_json", type=str, default="data/coco/train/annotations.json")
    p.add_argument("--image_dir", type=str, default="data/coco/train/images")
    p.add_argument("--output_dir", type=str, default="data/crops")
    p.add_argument("--iou_thresh", type=float, default=0.5, help="Min IoU to match YOLO pred to GT")
    p.add_argument("--conf_thresh", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument("--device", type=str, default="0")
    return p.parse_args()


def main():
    args = parse_args()

    from ultralytics import YOLO

    # Load COCO annotations
    print(f"Loading annotations from {args.coco_json}...")
    with open(args.coco_json) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    print(f"  {len(images)} images, {len(coco['annotations'])} annotations")

    # Load YOLO
    print(f"Loading YOLO from {args.yolo_weights}...")
    yolo = YOLO(args.yolo_weights)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir)

    # Load existing manifest to get max annotation_id for unique naming
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

    new_rows = []
    n_matched = 0
    n_unmatched = 0
    crop_id = 100000  # Start IDs high to avoid collisions with GT annotation IDs

    for img_idx, (img_id, img_info) in enumerate(sorted(images.items())):
        file_name = img_info["file_name"]
        img_path = image_dir / file_name

        # Run YOLO
        results = yolo.predict(
            source=str(img_path),
            conf=args.conf_thresh,
            imgsz=DETECTOR_IMGSZ,
            device=args.device,
            verbose=False,
        )

        if not results or len(results[0].boxes) == 0:
            continue

        # Read image for cropping
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # Get GT boxes for this image
        gt_anns = anns_by_image.get(img_id, [])
        gt_boxes = [coco_to_xyxy(ann["bbox"]) for ann in gt_anns]
        gt_matched = [False] * len(gt_anns)

        # Get YOLO predictions
        pred_boxes = results[0].boxes.xyxy.cpu().numpy()
        pred_confs = results[0].boxes.conf.cpu().numpy()

        # Match each YOLO pred to best GT by IoU
        for pred_idx in range(pred_boxes.shape[0]):
            pred_box = pred_boxes[pred_idx].tolist()

            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_matched[gt_idx]:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou < args.iou_thresh or best_gt_idx < 0:
                n_unmatched += 1
                continue

            gt_matched[best_gt_idx] = True
            gt_ann = gt_anns[best_gt_idx]
            cat_id = gt_ann["category_id"]

            # Crop using YOLO bbox with buffer (matches inference pipeline)
            x1, y1, x2, y2 = pred_box
            bw, bh = x2 - x1, y2 - y1
            pad_x = bw * CROP_BUFFER
            pad_y = bh * CROP_BUFFER
            cx1 = max(0, int(x1 - pad_x))
            cy1 = max(0, int(y1 - pad_y))
            cx2 = min(img_w, int(x2 + pad_x))
            cy2 = min(img_h, int(y2 + pad_y))

            crop = img[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            crop = resize_max_side(crop, CROP_SAVE_SIZE)

            # Save with unique name
            crop_name = f"yolo_{crop_id}_{cat_id}.jpg"
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
                    "image_id": img_id,
                    "source": "yolo_mined",
                }
            )
            n_matched += 1
            crop_id += 1

        if (img_idx + 1) % 50 == 0:
            print(f"  Processed {img_idx + 1}/{len(images)} images, {n_matched} mined crops so far")

    # Append new rows to manifest
    if new_rows:
        all_rows = existing_rows + new_rows
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "category_id", "annotation_id", "image_id", "source"])
            writer.writeheader()
            writer.writerows(all_rows)

    print(f"\nDone!")
    print(f"  Matched YOLO crops: {n_matched}")
    print(f"  Unmatched predictions: {n_unmatched}")
    print(f"  Total manifest entries: {len(existing_rows) + len(new_rows)}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
