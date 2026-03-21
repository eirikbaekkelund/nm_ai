"""Score prediction files against ground truth annotations.

Compares multiple prediction files (e.g., A/B/C ref strategies) on the same
detection boxes. Since detection is identical, focuses on classification accuracy.

Usage:
    python -m vision_task.score_predictions \
        predictions_A.json predictions_B.json predictions_C.json

    # Score on val images only:
    python -m vision_task.score_predictions predictions_A.json --val_only
"""

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
IOU_THRESHOLD = 0.5


def compute_iou_matrix(boxes_a, boxes_b):
    """IoU between [x1,y1,x2,y2] box arrays. Returns [A, B] matrix."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    return np.where(union > 0, inter / union, 0)


def match_predictions_to_gt(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Greedy IoU matching. Returns (matches, unmatched_preds, unmatched_gts)."""
    if len(pred_boxes) == 0:
        return [], [], list(range(len(gt_boxes)))
    if len(gt_boxes) == 0:
        return [], list(range(len(pred_boxes))), []

    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
    matches = []
    matched_preds = set()
    matched_gts = set()

    while True:
        if iou_matrix.size == 0:
            break
        max_iou = iou_matrix.max()
        if max_iou < iou_threshold:
            break
        pi, gi = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        matches.append((int(pi), int(gi), float(max_iou)))
        matched_preds.add(int(pi))
        matched_gts.add(int(gi))
        iou_matrix[pi, :] = 0
        iou_matrix[:, gi] = 0

    unmatched_preds = [i for i in range(len(pred_boxes)) if i not in matched_preds]
    unmatched_gts = [i for i in range(len(gt_boxes)) if i not in matched_gts]
    return matches, unmatched_preds, unmatched_gts


def coco_to_xyxy(bbox):
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def get_val_image_ids(annotations):
    """Reproduce the val split (seed=42, 15%) at the image level."""
    import random
    all_image_ids = sorted(img["id"] for img in annotations["images"])
    rng = random.Random(42)
    rng.shuffle(all_image_ids)
    n_val = int(len(all_image_ids) * 0.15)
    return set(all_image_ids[:n_val])


def score_file(pred_path, annotations, id_to_catname, val_image_ids=None):
    """Score one predictions file. Returns dict of metrics."""
    with open(pred_path) as f:
        predictions = json.load(f)

    # Group predictions and GT by image_id
    preds_by_image = defaultdict(list)
    for p in predictions:
        preds_by_image[p["image_id"]].append(p)

    gt_by_image = defaultdict(list)
    for ann in annotations["annotations"]:
        gt_by_image[ann["image_id"]].append(ann)

    # Determine which images to score
    if val_image_ids:
        image_ids = sorted(val_image_ids & (set(preds_by_image.keys()) | set(gt_by_image.keys())))
    else:
        image_ids = sorted(set(preds_by_image.keys()) | set(gt_by_image.keys()))

    # Accumulators
    total_tp = 0
    total_fp = 0
    total_fn = 0
    cls_correct = 0
    cls_total = 0
    per_cat_correct = Counter()
    per_cat_total = Counter()
    confusion_pairs = Counter()

    for img_id in image_ids:
        gt_anns = gt_by_image.get(img_id, [])
        preds = preds_by_image.get(img_id, [])

        gt_boxes = np.array([coco_to_xyxy(a["bbox"]) for a in gt_anns], dtype=np.float32) if gt_anns else np.empty((0, 4), dtype=np.float32)
        gt_cats = [a["category_id"] for a in gt_anns]

        pred_boxes = np.array([coco_to_xyxy(p["bbox"]) for p in preds], dtype=np.float32) if preds else np.empty((0, 4), dtype=np.float32)
        pred_cats = [p["category_id"] for p in preds]

        matches, unmatched_preds, unmatched_gts = match_predictions_to_gt(
            pred_boxes, gt_boxes, IOU_THRESHOLD
        )

        total_tp += len(matches)
        total_fp += len(unmatched_preds)
        total_fn += len(unmatched_gts)

        for pi, gi, iou in matches:
            gt_cat = gt_cats[gi]
            pred_cat = pred_cats[pi]
            cls_total += 1
            per_cat_total[gt_cat] += 1
            if pred_cat == gt_cat:
                cls_correct += 1
                per_cat_correct[gt_cat] += 1
            else:
                confusion_pairs[(gt_cat, pred_cat)] += 1

    recall = total_tp / max(total_tp + total_fn, 1)
    precision = total_tp / max(total_tp + total_fp, 1)
    cls_acc = cls_correct / max(cls_total, 1)
    n_errors = cls_total - cls_correct

    # Per-category worst
    per_cat_acc = []
    for cat_id in per_cat_total:
        total = per_cat_total[cat_id]
        correct = per_cat_correct.get(cat_id, 0)
        per_cat_acc.append({
            "cat": cat_id,
            "name": id_to_catname.get(cat_id, "?"),
            "total": total,
            "correct": correct,
            "errors": total - correct,
            "acc": correct / total,
        })
    per_cat_acc.sort(key=lambda x: (x["acc"], -x["total"]))

    return {
        "file": str(pred_path),
        "n_images": len(image_ids),
        "detection": {
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "recall": round(recall, 4), "precision": round(precision, 4),
        },
        "classification": {
            "correct": cls_correct, "total": cls_total,
            "accuracy": round(cls_acc, 4), "errors": n_errors,
        },
        "worst_categories": per_cat_acc[:20],
        "top_confusion": [
            {"gt": gt, "pred": pred, "count": cnt,
             "gt_name": id_to_catname.get(gt, "?")[:35],
             "pred_name": id_to_catname.get(pred, "?")[:35]}
            for (gt, pred), cnt in confusion_pairs.most_common(15)
        ],
    }


def parse_args():
    p = argparse.ArgumentParser(description="Score prediction files against GT")
    p.add_argument("predictions", nargs="+", help="Prediction JSON files to score")
    p.add_argument("--annotations", default="data/coco/train/annotations.json")
    p.add_argument("--val_only", action="store_true", help="Score on val split only (seed=42, 15%)")
    return p.parse_args()


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    with open(args.annotations) as f:
        annotations = json.load(f)
    id_to_catname = {c["id"]: c["name"] for c in annotations["categories"]}

    val_image_ids = get_val_image_ids(annotations) if args.val_only else None
    split_label = "VAL" if args.val_only else "ALL"

    results = []
    for pred_path in args.predictions:
        r = score_file(pred_path, annotations, id_to_catname, val_image_ids)
        results.append(r)

    # ── Comparison table ──
    logger.info("")
    logger.info("=" * 90)
    logger.info("PREDICTION SCORING (%s images)", split_label)
    logger.info("=" * 90)
    logger.info("")
    logger.info("%-40s  %6s  %6s  %6s  %8s  %6s", "File", "TP", "FP", "FN", "Recall", "Prec")
    logger.info("-" * 80)
    for r in results:
        d = r["detection"]
        logger.info("%-40s  %6d  %6d  %6d  %8.4f  %6.4f",
                     Path(r["file"]).name, d["tp"], d["fp"], d["fn"], d["recall"], d["precision"])

    logger.info("")
    logger.info("%-40s  %8s  %8s  %8s  %8s", "File", "Correct", "Total", "Errors", "Acc")
    logger.info("-" * 80)
    for r in results:
        c = r["classification"]
        logger.info("%-40s  %8d  %8d  %8d  %8.4f",
                     Path(r["file"]).name, c["correct"], c["total"], c["errors"], c["accuracy"])

    # ── Detailed per-file ──
    for r in results:
        logger.info("")
        logger.info("=" * 90)
        logger.info("DETAILS: %s", Path(r["file"]).name)
        logger.info("=" * 90)
        c = r["classification"]
        logger.info("  Classification: %d/%d = %.4f (%.1f%% acc, %d errors)",
                     c["correct"], c["total"], c["accuracy"], c["accuracy"] * 100, c["errors"])

        logger.info("")
        logger.info("  Worst categories (min samples ≥ 2):")
        shown = 0
        for cat in r["worst_categories"]:
            if cat["total"] < 2:
                continue
            logger.info("    cat=%3d  acc=%.2f  (%d/%d, %d err)  %s",
                         cat["cat"], cat["acc"], cat["correct"], cat["total"],
                         cat["errors"], cat["name"][:50])
            shown += 1
            if shown >= 15:
                break

        logger.info("")
        logger.info("  Top confusion pairs:")
        for cp in r["top_confusion"][:10]:
            logger.info("    %3d -> %3d  (%dx)  %s -> %s",
                         cp["gt"], cp["pred"], cp["count"], cp["gt_name"], cp["pred_name"])


if __name__ == "__main__":
    main()
