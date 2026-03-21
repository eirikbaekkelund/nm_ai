"""
Local inference — runs YOLO + DINOv2 classifier on shelf images using native models.

No ONNX export needed. Uses ultralytics YOLO + vision_task.embedder directly.
Outputs same COCO-format predictions JSON as the sandbox run.py.

Usage:
    # Predict on training images:
    python -m vision_task.predict

    # Predict + visualize:
    python -m vision_task.predict --visualize

    # Custom paths:
    python -m vision_task.predict \
        --image_dir data/coco/train/images \
        --yolo_weights models/yolo_best.pt \
        --classifier_weights models/classifier.pt \
        --output predictions.json \
        --visualize --vis_dir experiments/visualizations
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from vision_task.caqe import apply_caqe
from vision_task.config import CROP_BUFFER, DETECTOR_IMGSZ
from vision_task.data.transforms import get_eval_transform
from vision_task.embedder import GroceryEmbedder

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

UNKNOWN_CATEGORY_ID = 355
UNKNOWN_THRESHOLD = 0.0


def parse_args():
    p = argparse.ArgumentParser(description="Local inference (native YOLO + DINOv2)")
    p.add_argument("--image_dir", type=str, default="data/coco/train/images")
    p.add_argument("--yolo_weights", type=str, default="models/yolo_best.pt")
    p.add_argument(
        "--classifier_weights",
        type=str,
        default="models/classifier.pt",
        help="Training checkpoint or stripped state_dict",
    )
    p.add_argument("--ref_embeddings", type=str, default="models/ref_embeddings.pt")
    p.add_argument("--output", type=str, default="predictions.json")
    p.add_argument("--detect_conf", type=float, default=0.25)
    p.add_argument("--classify_batch", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--max_images", type=int, default=0, help="Max images to process (0=all)")
    # CAQE
    p.add_argument("--caqe", action="store_true", help="Enable Context-Aware Query Expansion")
    p.add_argument("--caqe_k", type=int, default=3, help="CAQE: number of spatial neighbors")
    p.add_argument("--caqe_alpha", type=float, default=0.5, help="CAQE: weight on original embedding")
    # Visualization
    p.add_argument("--visualize", action="store_true", help="Draw predictions on images")
    p.add_argument("--vis_dir", type=str, default="experiments/visualizations")
    p.add_argument("--score_thresh", type=float, default=0.1, help="Min score to draw")
    p.add_argument("--max_vis", type=int, default=10, help="Max images to visualize (0=all)")
    return p.parse_args()


def load_classifier(weights_path, device):
    """Load GroceryEmbedder from training checkpoint or stripped state_dict."""
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    model = GroceryEmbedder(weights_path=None, freeze_backbone=True)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model


def image_id_from_filename(filename):
    stem = Path(filename).stem
    digits = "".join(c for c in stem if c.isdigit())
    return int(digits) if digits else hash(stem) % 100000


@torch.no_grad()
def embed_crop_images(crop_images, model, device, batch_size=128):
    """Embed PIL crop images → [N, D] L2-normalized embeddings."""
    transform = get_eval_transform()
    all_embs = []
    for i in range(0, len(crop_images), batch_size):
        batch_pil = crop_images[i : i + batch_size]
        batch = torch.stack([transform(img) for img in batch_pil]).to(device)
        embs = model(batch)
        embs = F.normalize(embs.float(), dim=1)
        all_embs.append(embs)
    return torch.cat(all_embs, dim=0)


def match_to_refs(embeddings, ref_embs, ref_ids):
    """Match [N, D] embeddings to refs → (category_ids, cos_scores)."""
    sim = embeddings @ ref_embs.T
    best_scores, best_indices = sim.max(dim=1)
    # Vectorized: look up category_id for each best-matching ref
    matched_cats = ref_ids[best_indices]
    # Apply unknown threshold
    if UNKNOWN_THRESHOLD > 0:
        unknown_mask = best_scores < UNKNOWN_THRESHOLD
        matched_cats[unknown_mask] = UNKNOWN_CATEGORY_ID
    return matched_cats.tolist(), best_scores.tolist()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- Load YOLO ---
    from ultralytics import YOLO

    yolo_path = Path(args.yolo_weights)
    if not yolo_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {yolo_path}")
    logger.info("Loading YOLO from %s", yolo_path)
    yolo = YOLO(str(yolo_path))

    # --- Load classifier ---
    cls_path = Path(args.classifier_weights)
    if not cls_path.exists():
        raise FileNotFoundError(f"Classifier weights not found: {cls_path}")
    logger.info("Loading classifier from %s", cls_path)
    cls_model = load_classifier(str(cls_path), device)

    # --- Load reference embeddings ---
    ref_path = Path(args.ref_embeddings)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference embeddings not found: {ref_path}")
    ref_data = torch.load(str(ref_path), map_location=device, weights_only=True)
    ref_embs = F.normalize(ref_data["embeddings"].to(device).float(), dim=1)
    ref_ids = ref_data["category_ids"]
    logger.info("Reference embeddings: %d categories", len(ref_ids))

    # --- Collect images ---
    image_dir = Path(args.image_dir)
    image_paths = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    )
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    logger.info("Processing %d images from %s", len(image_paths), image_dir)

    # --- Inference loop ---
    predictions = []

    for img_path in image_paths:
        image_id = image_id_from_filename(img_path.name)
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # YOLO detection (native ultralytics handles letterbox internally)
        results = yolo.predict(
            source=str(img_path),
            conf=args.detect_conf,
            imgsz=DETECTOR_IMGSZ,
            device=device,
            verbose=False,
        )

        if not results or len(results[0].boxes) == 0:
            continue

        boxes = results[0].boxes
        det_boxes = boxes.xyxy.cpu()  # [N, 4] x1,y1,x2,y2
        det_scores = boxes.conf.cpu()  # [N]

        # Crop detections with 5% padding (matches training crops)
        crop_images = []
        valid_indices = []
        for i in range(det_boxes.shape[0]):
            x1, y1, x2, y2 = det_boxes[i].tolist()
            bw, bh = x2 - x1, y2 - y1
            pad_x = bw * CROP_BUFFER
            pad_y = bh * CROP_BUFFER
            x1 = max(0, int(x1 - pad_x))
            y1 = max(0, int(y1 - pad_y))
            x2 = min(orig_w, int(x2 + pad_x))
            y2 = min(orig_h, int(y2 + pad_y))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img.crop((x1, y1, x2, y2))
            crop_images.append(crop)
            valid_indices.append(i)

        if not crop_images:
            continue

        # Embed → CAQE → Match
        embeddings = embed_crop_images(crop_images, cls_model, device, args.classify_batch)

        if args.caqe and len(valid_indices) > 1:
            valid_boxes = det_boxes[valid_indices]  # [M, 4] xyxy
            embeddings = apply_caqe(embeddings, valid_boxes, k=args.caqe_k, alpha=args.caqe_alpha)

        cat_ids, cos_scores = match_to_refs(embeddings, ref_embs, ref_ids)

        for j, vi in enumerate(valid_indices):
            x1, y1, x2, y2 = det_boxes[vi].tolist()
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": cat_ids[j],
                    "bbox": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2 - x1, 2),
                        round(y2 - y1, 2),
                    ],
                    "score": round(float(det_scores[vi]) * cos_scores[j], 4),
                }
            )

    # --- Save predictions ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f)
    logger.info("Wrote %d predictions to %s", len(predictions), output_path)

    # --- Visualize ---
    if args.visualize:
        from vision_task.visualize import draw_predictions, load_category_names

        annotations_path = ROOT / "data" / "coco" / "train" / "annotations.json"
        category_names = load_category_names(str(annotations_path))

        vis_dir = Path(args.vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Group predictions by image_id
        from collections import defaultdict

        by_image = defaultdict(list)
        for pred in predictions:
            by_image[pred["image_id"]].append(pred)

        count = 0
        for img_path in image_paths:
            image_id = image_id_from_filename(img_path.name)
            if image_id not in by_image:
                continue
            if args.max_vis > 0 and count >= args.max_vis:
                break

            img = Image.open(img_path).convert("RGB")
            annotated = draw_predictions(img, by_image[image_id], category_names, args.score_thresh)
            out_path = vis_dir / f"vis_{img_path.name}"
            annotated.save(str(out_path), quality=90)
            n_drawn = len([p for p in by_image[image_id] if p["score"] >= args.score_thresh])
            logger.info("  %s: %d predictions drawn", out_path.name, n_drawn)
            count += 1

        logger.info("Saved %d visualizations to %s", count, vis_dir)


if __name__ == "__main__":
    main()
