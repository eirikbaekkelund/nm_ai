"""
Comprehensive failure analysis for detection + classification pipeline.

Runs YOLO detection on val images, matches predictions to ground truth via IoU,
then runs classification on matched crops. Outputs detailed per-image, per-category,
and per-size-bucket diagnostics to identify exactly where the pipeline fails.

Usage:
    python -m vision_task.diagnose
    python -m vision_task.diagnose --output_dir experiments/diagnostics
    python -m vision_task.diagnose --det_conf 0.15 --classifier_weights experiments/phase5b_unfreeze4/best.pt
"""

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from vision_task.config import CROP_BUFFER, DETECTOR_IMGSZ
from vision_task.data.transforms import get_eval_transform
from vision_task.predict import load_classifier

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
UNKNOWN_CATEGORY_ID = 355
IOU_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# IoU matching
# ---------------------------------------------------------------------------


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of [x1,y1,x2,y2] boxes. Returns [A, B] matrix."""
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
    """Greedy IoU matching: assign each prediction to at most one GT box.

    Returns:
        matches: list of (pred_idx, gt_idx, iou) for true positives
        unmatched_preds: list of pred_idx (false positives)
        unmatched_gts: list of gt_idx (false negatives / missed detections)
    """
    if len(pred_boxes) == 0:
        return [], [], list(range(len(gt_boxes)))
    if len(gt_boxes) == 0:
        return [], list(range(len(pred_boxes))), []

    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)

    matches = []
    matched_preds = set()
    matched_gts = set()

    # Greedy matching: highest IoU first
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


# ---------------------------------------------------------------------------
# Size bucketing
# ---------------------------------------------------------------------------

SIZE_BUCKETS = [
    ("tiny", 0, 32**2),  # < 32x32 pixels
    ("small", 32**2, 64**2),  # 32x32 - 64x64
    ("medium", 64**2, 128**2),  # 64x64 - 128x128
    ("large", 128**2, float("inf")),
]


def box_area(box):
    """Area of [x1, y1, x2, y2] box."""
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def coco_to_xyxy(bbox):
    """Convert COCO [x, y, w, h] to [x1, y1, x2, y2]."""
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def size_bucket(area):
    for name, lo, hi in SIZE_BUCKETS:
        if lo <= area < hi:
            return name
    return "large"


# ---------------------------------------------------------------------------
# Classification diagnosis
# ---------------------------------------------------------------------------


@torch.no_grad()
def classify_single_crop(crop_pil, model, ref_embs, ref_ids, device, transform):
    """Classify a single PIL crop. Returns (predicted_cat, cosine_sim, top5_cats, top5_sims)."""
    tensor = transform(crop_pil).unsqueeze(0).to(device)
    emb = model(tensor)
    emb = F.normalize(emb.float(), dim=1)

    sim = (emb @ ref_embs.T).squeeze(0)  # [N_ref]
    top5_vals, top5_idx = sim.topk(min(5, len(sim)))

    top5_cats = [ref_ids[idx].item() for idx in top5_idx]
    top5_sims = [v.item() for v in top5_vals]

    return top5_cats[0], top5_sims[0], top5_cats, top5_sims


@torch.no_grad()
def classify_batch(crop_pils, model, ref_embs, ref_ids, device, transform, batch_size=128):
    """Classify a batch of PIL crops. Returns list of per-crop result dicts."""
    results = []
    for i in range(0, len(crop_pils), batch_size):
        batch_pil = crop_pils[i : i + batch_size]
        batch = torch.stack([transform(img) for img in batch_pil]).to(device)
        embs = model(batch)
        embs = F.normalize(embs.float(), dim=1)

        sim = embs @ ref_embs.T  # [B, N_ref]

        for j in range(len(batch_pil)):
            top5_vals, top5_idx = sim[j].topk(min(5, sim.shape[1]))
            top5_cats = [ref_ids[idx].item() for idx in top5_idx]
            top5_sims = [v.item() for v in top5_vals]
            results.append(
                {
                    "pred_cat": top5_cats[0],
                    "pred_sim": top5_sims[0],
                    "top5_cats": top5_cats,
                    "top5_sims": top5_sims,
                }
            )
    return results


# ---------------------------------------------------------------------------
# Main diagnostic pipeline
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Pipeline failure analysis")
    p.add_argument("--yolo_weights", type=str, default="models/yolo_best.pt")
    p.add_argument("--classifier_weights", type=str, default="models/classifier_best.pt")
    p.add_argument("--ref_embeddings", type=str, default="models/ref_embeddings.pt")
    p.add_argument("--annotations", type=str, default="data/coco/train/annotations.json")
    p.add_argument("--val_dir", type=str, default="data/shelf_yolo/images/val")
    p.add_argument("--image_dir", type=str, default="data/coco/train/images")
    p.add_argument("--output_dir", type=str, default="experiments/diagnostics")
    p.add_argument("--det_conf", type=float, default=0.25)
    p.add_argument("--iou_threshold", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=128)
    return p.parse_args()


def main():
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load annotations ---
    with open(args.annotations, "r") as f:
        coco_data = json.load(f)

    id_to_image = {img["id"]: img for img in coco_data["images"]}
    id_to_catname = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    # --- Identify val images ---
    val_dir = Path(args.val_dir)
    name_to_id = {img["file_name"]: img["id"] for img in coco_data["images"]}
    val_image_ids = set()
    val_filenames = {}
    for p in sorted(val_dir.iterdir()):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.name in name_to_id:
            img_id = name_to_id[p.name]
            val_image_ids.add(img_id)
            val_filenames[img_id] = p.name

    logger.info("Val images: %d", len(val_image_ids))

    # Group GT annotations by image_id (val only)
    gt_by_image = defaultdict(list)
    for ann in coco_data["annotations"]:
        if ann["image_id"] in val_image_ids:
            gt_by_image[ann["image_id"]].append(ann)

    total_gt_anns = sum(len(v) for v in gt_by_image.values())
    logger.info("Val GT annotations: %d", total_gt_anns)

    # --- Load YOLO ---
    from ultralytics import YOLO

    logger.info("Loading YOLO from %s", args.yolo_weights)
    yolo = YOLO(args.yolo_weights)

    # --- Load classifier ---
    logger.info("Loading classifier from %s", args.classifier_weights)
    cls_model = load_classifier(args.classifier_weights, device)

    # --- Load reference embeddings ---
    ref_data = torch.load(args.ref_embeddings, map_location=device, weights_only=True)
    ref_embs = F.normalize(ref_data["embeddings"].to(device).float(), dim=1)
    ref_ids = ref_data["category_ids"]
    logger.info("Reference embeddings: %s", ref_embs.shape)

    transform = get_eval_transform()

    # --- Per-image analysis ---
    image_dir = Path(args.image_dir)

    # Accumulators
    per_image_results = []

    # Detection stats
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Size-based detection stats
    size_stats = {name: {"tp": 0, "fn": 0, "total": 0} for name, _, _ in SIZE_BUCKETS}

    # Classification stats
    cls_correct = 0
    cls_total = 0
    per_cat_correct = Counter()
    per_cat_total = Counter()
    confusion_pairs = Counter()  # (gt_cat, pred_cat) -> count
    misclassification_details = []

    # Cosine similarity distributions
    sim_correct = []
    sim_incorrect = []
    sim_all = []

    # FN details (missed detections)
    fn_details = []

    # FP details
    fp_details = []

    for img_id in sorted(val_image_ids):
        fname = val_filenames[img_id]
        img_path = image_dir / fname
        if not img_path.exists():
            logger.warning("Image not found: %s", img_path)
            continue

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        gt_anns = gt_by_image.get(img_id, [])
        gt_boxes = np.array([coco_to_xyxy(ann["bbox"]) for ann in gt_anns], dtype=np.float32)
        gt_cats = [ann["category_id"] for ann in gt_anns]
        gt_areas = [box_area(b) for b in gt_boxes]

        # --- Run YOLO ---
        results = yolo.predict(
            source=str(img_path),
            conf=args.det_conf,
            imgsz=DETECTOR_IMGSZ,
            device=device,
            verbose=False,
        )

        if results and len(results[0].boxes) > 0:
            pred_boxes = results[0].boxes.xyxy.cpu().numpy()
            pred_scores = results[0].boxes.conf.cpu().numpy()
        else:
            pred_boxes = np.empty((0, 4), dtype=np.float32)
            pred_scores = np.empty((0,), dtype=np.float32)

        # --- Match predictions to GT ---
        matches, unmatched_preds, unmatched_gts = match_predictions_to_gt(pred_boxes, gt_boxes, args.iou_threshold)

        n_tp = len(matches)
        n_fp = len(unmatched_preds)
        n_fn = len(unmatched_gts)
        total_tp += n_tp
        total_fp += n_fp
        total_fn += n_fn

        # Size-based FN analysis
        for gi in unmatched_gts:
            area = gt_areas[gi]
            bucket = size_bucket(area)
            size_stats[bucket]["fn"] += 1
            fn_details.append(
                {
                    "image_id": img_id,
                    "filename": fname,
                    "gt_ann_id": gt_anns[gi]["id"],
                    "category_id": gt_cats[gi],
                    "category_name": id_to_catname.get(gt_cats[gi], "?"),
                    "bbox_xywh": gt_anns[gi]["bbox"],
                    "area": float(area),
                    "size_bucket": bucket,
                }
            )

        # Count total GT and TP per size bucket
        for gi in range(len(gt_anns)):
            bucket = size_bucket(gt_areas[gi])
            size_stats[bucket]["total"] += 1
        for _, gi, _ in matches:
            bucket = size_bucket(gt_areas[gi])
            size_stats[bucket]["tp"] += 1

        # FP details
        for pi in unmatched_preds:
            fp_details.append(
                {
                    "image_id": img_id,
                    "filename": fname,
                    "pred_box": pred_boxes[pi].tolist(),
                    "pred_score": float(pred_scores[pi]),
                    "area": float(box_area(pred_boxes[pi])),
                }
            )

        # --- Classify matched detections ---
        if matches:
            # Crop matched predictions
            matched_crops = []
            matched_gt_cats = []
            matched_ious = []
            matched_pred_scores = []

            for pi, gi, iou in matches:
                box = pred_boxes[pi]
                bw, bh = box[2] - box[0], box[3] - box[1]
                pad_x = bw * CROP_BUFFER
                pad_y = bh * CROP_BUFFER
                x1 = max(0, int(box[0] - pad_x))
                y1 = max(0, int(box[1] - pad_y))
                x2 = min(img_w, int(box[2] + pad_x))
                y2 = min(img_h, int(box[3] + pad_y))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = img.crop((x1, y1, x2, y2))
                matched_crops.append(crop)
                matched_gt_cats.append(gt_cats[gi])
                matched_ious.append(iou)
                matched_pred_scores.append(float(pred_scores[pi]))

            if matched_crops:
                cls_results = classify_batch(
                    matched_crops, cls_model, ref_embs, ref_ids, device, transform, args.batch_size
                )

                for j, cr in enumerate(cls_results):
                    gt_cat = matched_gt_cats[j]
                    pred_cat = cr["pred_cat"]
                    sim_score = cr["pred_sim"]

                    cls_total += 1
                    per_cat_total[gt_cat] += 1
                    sim_all.append(sim_score)

                    if pred_cat == gt_cat:
                        cls_correct += 1
                        per_cat_correct[gt_cat] += 1
                        sim_correct.append(sim_score)
                    else:
                        sim_incorrect.append(sim_score)
                        confusion_pairs[(gt_cat, pred_cat)] += 1
                        misclassification_details.append(
                            {
                                "image_id": img_id,
                                "filename": fname,
                                "gt_cat": gt_cat,
                                "gt_name": id_to_catname.get(gt_cat, "?"),
                                "pred_cat": pred_cat,
                                "pred_name": id_to_catname.get(pred_cat, "?"),
                                "cosine_sim": round(sim_score, 4),
                                "top5_cats": cr["top5_cats"],
                                "top5_sims": [round(s, 4) for s in cr["top5_sims"]],
                                "iou": round(matched_ious[j], 3),
                                "det_score": round(matched_pred_scores[j], 4),
                            }
                        )

        img_result = {
            "image_id": img_id,
            "filename": fname,
            "n_gt": len(gt_anns),
            "n_pred": len(pred_boxes),
            "tp": n_tp,
            "fp": n_fp,
            "fn": n_fn,
            "recall": n_tp / max(len(gt_anns), 1),
            "precision": n_tp / max(len(pred_boxes), 1),
        }
        per_image_results.append(img_result)
        logger.info(
            "  %s: GT=%d Pred=%d TP=%d FP=%d FN=%d recall=%.2f",
            fname,
            len(gt_anns),
            len(pred_boxes),
            n_tp,
            n_fp,
            n_fn,
            img_result["recall"],
        )

    # --- Aggregate and write reports ---
    logger.info("=" * 60)
    logger.info("DETECTION SUMMARY")
    logger.info("=" * 60)
    recall = total_tp / max(total_tp + total_fn, 1)
    precision = total_tp / max(total_tp + total_fp, 1)
    logger.info("TP=%d  FP=%d  FN=%d", total_tp, total_fp, total_fn)
    logger.info(
        "Recall=%.4f  Precision=%.4f  F1=%.4f",
        recall,
        precision,
        2 * recall * precision / max(recall + precision, 1e-8),
    )

    logger.info("\nDetection by size bucket:")
    for name, _, _ in SIZE_BUCKETS:
        s = size_stats[name]
        r = s["tp"] / max(s["total"], 1)
        logger.info("  %-8s: total=%4d  tp=%4d  fn=%4d  recall=%.3f", name, s["total"], s["tp"], s["fn"], r)

    logger.info("\n" + "=" * 60)
    logger.info("CLASSIFICATION SUMMARY")
    logger.info("=" * 60)
    cls_acc = cls_correct / max(cls_total, 1)
    logger.info("Accuracy (on matched detections): %d/%d = %.4f", cls_correct, cls_total, cls_acc)

    if sim_correct:
        logger.info(
            "Cosine sim (correct):   mean=%.4f  std=%.4f  min=%.4f",
            np.mean(sim_correct),
            np.std(sim_correct),
            np.min(sim_correct),
        )
    if sim_incorrect:
        logger.info(
            "Cosine sim (incorrect): mean=%.4f  std=%.4f  max=%.4f",
            np.mean(sim_incorrect),
            np.std(sim_incorrect),
            np.max(sim_incorrect),
        )

    # Per-category accuracy (worst categories)
    per_cat_acc = {}
    for cat_id in per_cat_total:
        total = per_cat_total[cat_id]
        correct = per_cat_correct.get(cat_id, 0)
        per_cat_acc[cat_id] = {
            "category_id": cat_id,
            "name": id_to_catname.get(cat_id, "?"),
            "total": total,
            "correct": correct,
            "accuracy": correct / total,
        }

    worst_cats = sorted(per_cat_acc.values(), key=lambda x: (x["accuracy"], -x["total"]))
    logger.info("\nWorst 20 categories by accuracy (min 2 samples):")
    shown = 0
    for cat in worst_cats:
        if cat["total"] < 2:
            continue
        logger.info(
            "  cat=%3d  acc=%.2f  (%d/%d)  %s",
            cat["category_id"],
            cat["accuracy"],
            cat["correct"],
            cat["total"],
            cat["name"],
        )
        shown += 1
        if shown >= 20:
            break

    # Top confusion pairs
    logger.info("\nTop 20 confusion pairs (gt -> pred):")
    for (gt_cat, pred_cat), count in confusion_pairs.most_common(20):
        gt_name = id_to_catname.get(gt_cat, "?")[:40]
        pred_name = id_to_catname.get(pred_cat, "?")[:40]
        logger.info("  %3d -> %3d  (%dx)  %s -> %s", gt_cat, pred_cat, count, gt_name, pred_name)

    # Images with most missed detections
    worst_images = sorted(per_image_results, key=lambda x: x["fn"], reverse=True)
    logger.info("\nImages with most missed detections (FN):")
    for img in worst_images[:10]:
        logger.info("  %s: FN=%d (of %d GT), recall=%.2f", img["filename"], img["fn"], img["n_gt"], img["recall"])

    # --- Save detailed JSON report ---
    report = {
        "config": {
            "yolo_weights": args.yolo_weights,
            "classifier_weights": args.classifier_weights,
            "ref_embeddings": args.ref_embeddings,
            "det_conf": args.det_conf,
            "iou_threshold": args.iou_threshold,
        },
        "detection_summary": {
            "total_gt": total_tp + total_fn,
            "total_pred": total_tp + total_fp,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(2 * recall * precision / max(recall + precision, 1e-8), 4),
        },
        "detection_by_size": {
            name: {
                "total": size_stats[name]["total"],
                "tp": size_stats[name]["tp"],
                "fn": size_stats[name]["fn"],
                "recall": round(size_stats[name]["tp"] / max(size_stats[name]["total"], 1), 4),
            }
            for name, _, _ in SIZE_BUCKETS
        },
        "classification_summary": {
            "total_matched": cls_total,
            "correct": cls_correct,
            "accuracy": round(cls_acc, 4),
            "sim_correct_mean": round(float(np.mean(sim_correct)), 4) if sim_correct else None,
            "sim_correct_std": round(float(np.std(sim_correct)), 4) if sim_correct else None,
            "sim_incorrect_mean": round(float(np.mean(sim_incorrect)), 4) if sim_incorrect else None,
            "sim_incorrect_std": round(float(np.std(sim_incorrect)), 4) if sim_incorrect else None,
        },
        "per_image": per_image_results,
        "per_category_accuracy": sorted(
            list(per_cat_acc.values()),
            key=lambda x: (x["accuracy"], -x["total"]),
        ),
        "top_confusion_pairs": [
            {
                "gt_cat": gt_cat,
                "gt_name": id_to_catname.get(gt_cat, "?"),
                "pred_cat": pred_cat,
                "pred_name": id_to_catname.get(pred_cat, "?"),
                "count": count,
            }
            for (gt_cat, pred_cat), count in confusion_pairs.most_common(50)
        ],
        "misclassification_details": sorted(misclassification_details, key=lambda x: x["cosine_sim"], reverse=True,)[
            :200
        ],  # Top 200 most confident misclassifications
        "missed_detections": sorted(fn_details, key=lambda x: -x["area"],)[
            :200
        ],  # Largest missed detections first
        "false_positives": sorted(fp_details, key=lambda x: -x["pred_score"],)[
            :100
        ],  # Highest confidence FPs
        "cosine_sim_histogram": {
            "correct": sorted(sim_correct),
            "incorrect": sorted(sim_incorrect),
        },
    }

    report_path = output_dir / "diagnosis_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("\nFull report saved to %s", report_path)

    # --- Write human-readable summary ---
    summary_path = output_dir / "diagnosis_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("PIPELINE FAILURE ANALYSIS\n")
        f.write(f"  YOLO: {args.yolo_weights}\n")
        f.write(f"  Classifier: {args.classifier_weights}\n")
        f.write(f"  det_conf={args.det_conf}  iou_threshold={args.iou_threshold}\n")
        f.write("=" * 70 + "\n\n")

        f.write("DETECTION\n")
        f.write("-" * 40 + "\n")
        f.write(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}\n")
        f.write(f"  Recall={recall:.4f}  Precision={precision:.4f}\n\n")

        f.write("  By size bucket:\n")
        for name, _, _ in SIZE_BUCKETS:
            s = size_stats[name]
            r = s["tp"] / max(s["total"], 1)
            f.write(f"    {name:8s}: {s['total']:4d} total, {s['tp']:4d} TP, {s['fn']:4d} FN, recall={r:.3f}\n")

        f.write(f"\n  Worst images (by FN):\n")
        for img in worst_images[:10]:
            f.write(f"    {img['filename']}: FN={img['fn']}/{img['n_gt']} recall={img['recall']:.2f}\n")

        f.write(f"\nCLASSIFICATION (on {cls_total} matched detections)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Accuracy: {cls_correct}/{cls_total} = {cls_acc:.4f}\n")
        if sim_correct:
            f.write(f"  Cosine sim (correct):   mean={np.mean(sim_correct):.4f} std={np.std(sim_correct):.4f}\n")
        if sim_incorrect:
            f.write(f"  Cosine sim (incorrect): mean={np.mean(sim_incorrect):.4f} std={np.std(sim_incorrect):.4f}\n")

        f.write(f"\n  Worst categories (min 2 samples):\n")
        shown = 0
        for cat in worst_cats:
            if cat["total"] < 2:
                continue
            f.write(
                f"    cat={cat['category_id']:3d}  acc={cat['accuracy']:.2f}  "
                f"({cat['correct']}/{cat['total']})  {cat['name']}\n"
            )
            shown += 1
            if shown >= 30:
                break

        f.write(f"\n  Top confusion pairs:\n")
        for (gt_cat, pred_cat), count in confusion_pairs.most_common(30):
            gt_name = id_to_catname.get(gt_cat, "?")[:35]
            pred_name = id_to_catname.get(pred_cat, "?")[:35]
            f.write(f"    {gt_cat:3d} -> {pred_cat:3d}  ({count}x)  {gt_name} -> {pred_name}\n")

    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
