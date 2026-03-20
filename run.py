"""
Competition submission entry point.

Detects grocery products on shelf images (YOLO11x) and classifies them
(DINOv2 + cosine similarity to reference embeddings).

Usage (sandbox):
    python run.py --input_dir /path/to/test_images --output_file predictions.json

Output: COCO-format predictions JSON:
    [{"image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float}, ...]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from vision_task.config import (
    CLASSIFIER_RESIZE,
    CLASSIFIER_SIZE,
    DETECTOR_IMGSZ,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from vision_task.embedder import GroceryEmbedder

# --- Paths (relative to this file, bundled in .zip) ---
ROOT = Path(__file__).resolve().parent
YOLO_WEIGHTS = ROOT / "models" / "yolo_best.pt"
CLASSIFIER_CHECKPOINT = ROOT / "models" / "classifier_best.pt"
DINOV2_WEIGHTS = ROOT / "models" / "dinov2_vitb14.pth"
REF_EMBEDDINGS = ROOT / "models" / "ref_embeddings.pt"

# --- Thresholds (tune in Phase 8) ---
DETECT_CONF = 0.25  # Low to maximize recall (70% of score)
CLASSIFY_BATCH = 64
UNKNOWN_CATEGORY_ID = 355  # category for unknown_product
UNKNOWN_THRESHOLD = 0.0  # cosine sim below this → unknown (disabled by default)


def parse_args():
    p = argparse.ArgumentParser(description="NorgesGruppen shelf product detection")
    p.add_argument("--input_dir", type=str, required=True,
                    help="Directory containing shelf images")
    p.add_argument("--output_file", type=str, default="predictions.json",
                    help="Output COCO predictions JSON")
    p.add_argument("--detect_conf", type=float, default=DETECT_CONF)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def load_detector():
    """Load YOLO11x detector."""
    assert YOLO_WEIGHTS.exists(), f"YOLO weights not found: {YOLO_WEIGHTS}"
    model = YOLO(str(YOLO_WEIGHTS))
    return model


def load_classifier(device):
    """Load DINOv2 embedder from fine-tuned checkpoint."""
    assert DINOV2_WEIGHTS.exists(), f"DINOv2 weights not found: {DINOV2_WEIGHTS}"
    assert CLASSIFIER_CHECKPOINT.exists(), f"Classifier checkpoint not found: {CLASSIFIER_CHECKPOINT}"

    model = GroceryEmbedder(
        weights_path=str(DINOV2_WEIGHTS),
        freeze_backbone=True,
    )
    checkpoint = torch.load(CLASSIFIER_CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def load_ref_embeddings(device):
    """Load pre-computed reference embeddings."""
    assert REF_EMBEDDINGS.exists(), f"Ref embeddings not found: {REF_EMBEDDINGS}"
    data = torch.load(REF_EMBEDDINGS, map_location=device, weights_only=True)
    ref_embs = data["embeddings"].to(device)  # [C, 768] L2-normalized
    ref_ids = data["category_ids"]  # [C] int64
    return ref_embs, ref_ids


def get_eval_transform():
    """Eval transform: Resize(546) → CenterCrop(518) → Normalize."""
    return transforms.Compose([
        transforms.Resize(CLASSIFIER_RESIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(CLASSIFIER_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def extract_crops(image, boxes):
    """Crop detected regions from PIL image.

    Args:
        image: PIL Image
        boxes: tensor [N, 4] in xyxy format (pixels)

    Returns:
        list of PIL Image crops
    """
    w, h = image.size
    crops = []
    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        # Clamp to image bounds
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        crops.append(image.crop((x1, y1, x2, y2)))
    return crops


@torch.no_grad()
def classify_crops(crops, model, transform, ref_embs, ref_ids, device):
    """Classify a list of PIL image crops via cosine similarity.

    Returns:
        category_ids: list[int]
        scores: list[float] (cosine similarity to best match)
    """
    if not crops:
        return [], []

    category_ids = []
    cos_scores = []

    for i in range(0, len(crops), CLASSIFY_BATCH):
        batch_crops = crops[i:i + CLASSIFY_BATCH]
        tensors = torch.stack([transform(c) for c in batch_crops]).to(device)

        embs = model(tensors)  # [B, 768]
        embs = F.normalize(embs.float(), dim=1)

        # Cosine similarity: [B, C]
        sim = embs @ ref_embs.T
        best_scores, best_indices = sim.max(dim=1)

        for j in range(len(batch_crops)):
            score = best_scores[j].item()
            if score < UNKNOWN_THRESHOLD:
                category_ids.append(UNKNOWN_CATEGORY_ID)
            else:
                category_ids.append(ref_ids[best_indices[j]].item())
            cos_scores.append(score)

    return category_ids, cos_scores


def image_id_from_filename(filename):
    """Extract numeric image_id from filename like 'img_00042.jpg' → 42."""
    stem = Path(filename).stem
    digits = "".join(c for c in stem if c.isdigit())
    return int(digits) if digits else hash(stem) % 100000


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load models ---
    t0 = time.time()
    detector = load_detector()
    classifier = load_classifier(device)
    ref_embs, ref_ids = load_ref_embeddings(device)
    transform = get_eval_transform()
    print(f"Models loaded in {time.time() - t0:.1f}s")
    print(f"  Ref embeddings: {ref_embs.shape} ({len(ref_ids)} categories)")

    if torch.cuda.is_available():
        print(f"  VRAM after loading: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    # --- Find images ---
    input_dir = Path(args.input_dir)
    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    )
    print(f"Found {len(image_paths)} images in {input_dir}")

    # --- Run inference ---
    predictions = []
    t_start = time.time()

    for img_idx, img_path in enumerate(image_paths):
        image = Image.open(img_path).convert("RGB")
        image_id = image_id_from_filename(img_path.name)

        # Stage 1: Detect
        results = detector(
            img_path,
            imgsz=DETECTOR_IMGSZ,
            conf=args.detect_conf,
            device=device,
            verbose=False,
        )
        boxes = results[0].boxes

        if len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.cpu()  # [N, 4]
        det_scores = boxes.conf.cpu()  # [N]

        # Stage 2: Crop & Classify
        crops = extract_crops(image, xyxy)
        if not crops:
            continue

        cat_ids, cos_scores = classify_crops(
            crops, classifier, transform, ref_embs, ref_ids, device
        )

        # Build COCO predictions
        for j in range(len(crops)):
            x1, y1, x2, y2 = xyxy[j].tolist()
            predictions.append({
                "image_id": image_id,
                "category_id": cat_ids[j],
                "bbox": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2 - x1, 2),
                    round(y2 - y1, 2),
                ],
                "score": round(float(det_scores[j]) * cos_scores[j], 4),
            })

        if (img_idx + 1) % 10 == 0 or img_idx == 0:
            elapsed = time.time() - t_start
            print(f"  [{img_idx + 1}/{len(image_paths)}] "
                  f"{len(boxes)} detections, {elapsed:.1f}s elapsed")

    # --- Free VRAM between stages if needed ---
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    # --- Save ---
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f)

    total_time = time.time() - t_start
    print(f"\nDone! {len(predictions)} predictions from {len(image_paths)} images")
    print(f"Saved to: {output_path}")
    print(f"Total inference time: {total_time:.1f}s "
          f"({total_time / max(len(image_paths), 1):.2f}s/image)")


if __name__ == "__main__":
    main()
