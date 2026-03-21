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

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import center_crop, resize

from ensemble_boxes import weighted_boxes_fusion

from vision_task.caqe import apply_caqe
from vision_task.config import (
    CLASSIFIER_RESIZE,
    CLASSIFIER_SIZE,
    CROP_BUFFER,
    DETECTOR_IMGSZ,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from vision_task.embedder import GroceryEmbedder

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

UNKNOWN_CATEGORY_ID = 355
UNKNOWN_THRESHOLD = 0.0

# SAHI defaults
TILE_SIZE = 640
TILE_OVERLAP = 0.25
MIN_DIM_FOR_TILING = 2000
WBF_IOU_THR = 0.55


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
    # SAHI
    p.add_argument("--sahi", action="store_true", help="Enable SAHI tiled inference")
    p.add_argument("--tile_size", type=int, default=TILE_SIZE, help="SAHI tile size")
    p.add_argument("--tile_overlap", type=float, default=TILE_OVERLAP, help="SAHI tile overlap ratio")
    p.add_argument("--wbf_iou", type=float, default=WBF_IOU_THR, help="WBF IoU threshold")
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


def extract_and_transform_crops(img_tensor, boxes, device):
    """Crop from GPU tensor, resize, center-crop, normalize — all on GPU.

    Args:
        img_tensor: [3, H, W] float32 tensor (0-1 range) on device
        boxes: [N, 4] x1,y1,x2,y2 detection boxes
        device: torch device

    Returns:
        (batch_tensor [M, 3, 518, 518] normalized, valid_indices list)
    """
    _, h, w = img_tensor.shape
    norm_mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    norm_std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    crops = []
    valid_indices = []
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i].tolist()
        bw, bh = x2 - x1, y2 - y1
        pad_x = bw * CROP_BUFFER
        pad_y = bh * CROP_BUFFER
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(w, int(x2 + pad_x))
        cy2 = min(h, int(y2 + pad_y))
        if cx2 <= cx1 or cy2 <= cy1:
            continue

        crop = img_tensor[:, cy1:cy2, cx1:cx2]
        crop = resize(
            crop,
            [CLASSIFIER_RESIZE],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        crop = center_crop(crop, [CLASSIFIER_SIZE, CLASSIFIER_SIZE])
        crops.append(crop)
        valid_indices.append(i)

    if not crops:
        return None, []

    batch = torch.stack(crops)
    batch = (batch - norm_mean) / norm_std
    return batch, valid_indices


@torch.no_grad()
def embed_crops(crop_tensors, model, device, batch_size=128):
    """Embed pre-transformed crop tensors → [N, D] L2-normalized embeddings."""
    all_embs = []
    n = crop_tensors.shape[0]
    for i in range(0, n, batch_size):
        batch = crop_tensors[i : i + batch_size].to(device)
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


def generate_tiles(img_h, img_w, tile_size=TILE_SIZE, overlap=TILE_OVERLAP):
    """Generate (x1, y1, x2, y2) tile coordinates with overlap."""
    stride = int(tile_size * (1 - overlap))
    tiles = []
    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            x2 = min(x + tile_size, img_w)
            y2 = min(y + tile_size, img_h)
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)
            tiles.append((x1, y1, x2, y2))
    return list(dict.fromkeys(tiles))


def detect_with_sahi(yolo, img_path, img_np, orig_h, orig_w, device, args):
    """Full-image + tiled SAHI detection via ultralytics, merged with WBF."""
    # Pass 1: full-image
    results = yolo.predict(
        source=str(img_path), conf=args.detect_conf, imgsz=DETECTOR_IMGSZ,
        device=device, half=True, verbose=False,
    )
    if results and len(results[0].boxes) > 0:
        full_boxes = results[0].boxes.xyxy.cpu().numpy()
        full_scores = results[0].boxes.conf.cpu().numpy()
    else:
        full_boxes = np.empty((0, 4), dtype=np.float32)
        full_scores = np.empty((0,), dtype=np.float32)

    # Skip tiling for small images
    if max(orig_h, orig_w) <= MIN_DIM_FOR_TILING:
        return (torch.from_numpy(full_boxes).float(),
                torch.from_numpy(full_scores).float())

    # Pass 2: tiled detection
    tiles = generate_tiles(orig_h, orig_w, args.tile_size, args.tile_overlap)
    all_boxes = [full_boxes]
    all_scores = [full_scores]

    for tx1, ty1, tx2, ty2 in tiles:
        tile_np = img_np[ty1:ty2, tx1:tx2]
        tres = yolo.predict(
            source=tile_np, conf=args.detect_conf, imgsz=DETECTOR_IMGSZ,
            device=device, half=True, verbose=False,
        )
        if tres and len(tres[0].boxes) > 0:
            tboxes = tres[0].boxes.xyxy.cpu().numpy()
            tscores = tres[0].boxes.conf.cpu().numpy()
            # Remap tile-local coords to original image coords
            tboxes[:, [0, 2]] += tx1
            tboxes[:, [1, 3]] += ty1
            tboxes[:, [0, 2]] = np.clip(tboxes[:, [0, 2]], 0, orig_w)
            tboxes[:, [1, 3]] = np.clip(tboxes[:, [1, 3]], 0, orig_h)
            all_boxes.append(tboxes)
            all_scores.append(tscores)
        else:
            all_boxes.append(np.empty((0, 4), dtype=np.float32))
            all_scores.append(np.empty((0,), dtype=np.float32))

    # Merge via WBF (expects [0,1]-normalized coords)
    boxes_list, scores_list, labels_list = [], [], []
    for b, s in zip(all_boxes, all_scores):
        if len(b) == 0:
            boxes_list.append(np.empty((0, 4), dtype=np.float32))
            scores_list.append(np.empty((0,), dtype=np.float32))
            labels_list.append(np.empty((0,), dtype=np.float32))
            continue
        nb = b.copy().astype(np.float32)
        nb[:, [0, 2]] /= orig_w
        nb[:, [1, 3]] /= orig_h
        nb = np.clip(nb, 0, 1)
        boxes_list.append(nb)
        scores_list.append(s.astype(np.float32))
        labels_list.append(np.zeros(len(s), dtype=np.float32))

    if all(len(b) == 0 for b in boxes_list):
        return (torch.empty((0, 4)), torch.empty((0,)))

    fb, fs, _ = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        iou_thr=args.wbf_iou, skip_box_thr=0.0,
    )
    fb[:, [0, 2]] *= orig_w
    fb[:, [1, 3]] *= orig_h
    return (torch.from_numpy(fb).float(), torch.from_numpy(fs).float())


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
    import time
    t_start = time.time()
    total_detections = 0
    total_crops = 0

    for img_idx, img_path in enumerate(image_paths):
        t_img = time.time()
        image_id = image_id_from_filename(img_path.name)
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # YOLO detection
        t_det = time.time()
        img_np_arr = np.array(img)

        if args.sahi:
            det_boxes, det_scores = detect_with_sahi(
                yolo, img_path, img_np_arr, orig_h, orig_w, device, args
            )
        else:
            results = yolo.predict(
                source=str(img_path),
                conf=args.detect_conf,
                imgsz=DETECTOR_IMGSZ,
                device=device,
                half=True,
                verbose=False,
            )
            if not results or len(results[0].boxes) == 0:
                det_boxes = torch.empty((0, 4))
                det_scores = torch.empty((0,))
            else:
                det_boxes = results[0].boxes.xyxy.cpu()
                det_scores = results[0].boxes.conf.cpu()
        dt_det = time.time() - t_det

        if det_boxes.shape[0] == 0:
            if (img_idx + 1) % 50 == 0 or img_idx == 0:
                logger.info(
                    "  [%d/%d] %s: 0 detections (det=%.2fs)",
                    img_idx + 1, len(image_paths), img_path.name, dt_det,
                )
            continue

        total_detections += det_boxes.shape[0]

        # GPU-side crop + transform (same approach as run.py)
        t_crop = time.time()
        img_tensor = torch.from_numpy(img_np_arr).permute(2, 0, 1).to(
            device=device, dtype=torch.float32
        ).div_(255.0)
        crop_tensors, valid_indices = extract_and_transform_crops(
            img_tensor, det_boxes, device
        )
        del img_tensor
        dt_crop = time.time() - t_crop

        if crop_tensors is None:
            continue

        total_crops += crop_tensors.shape[0]

        # Embed → CAQE → Match
        t_cls = time.time()
        embeddings = embed_crops(crop_tensors, cls_model, device, args.classify_batch)
        del crop_tensors

        if args.caqe and len(valid_indices) > 1:
            valid_boxes = det_boxes[valid_indices]  # [M, 4] xyxy
            embeddings = apply_caqe(embeddings, valid_boxes, k=args.caqe_k, alpha=args.caqe_alpha)

        cat_ids, cos_scores = match_to_refs(embeddings, ref_embs, ref_ids)
        dt_cls = time.time() - t_cls

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

        if (img_idx + 1) % 25 == 0 or img_idx == 0:
            elapsed = time.time() - t_start
            rate = (img_idx + 1) / elapsed
            eta = (len(image_paths) - img_idx - 1) / rate if rate > 0 else 0
            logger.info(
                "  [%d/%d] %d dets, %d crops | det=%.2fs crop=%.2fs cls=%.2fs | "
                "%.1f img/s, ETA %.0fs",
                img_idx + 1, len(image_paths),
                det_boxes.shape[0], len(valid_indices),
                dt_det, dt_crop, dt_cls,
                rate, eta,
            )

    elapsed_total = time.time() - t_start
    logger.info(
        "Inference complete: %d images, %d detections, %d crops in %.1fs (%.1f img/s)",
        len(image_paths), total_detections, total_crops, elapsed_total,
        len(image_paths) / elapsed_total if elapsed_total > 0 else 0,
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
