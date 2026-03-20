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
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import resize, center_crop, normalize, to_tensor
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
CLASSIFY_BATCH = 128  # H100 can handle more; L4 can do 64 easily at 518
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


def load_detector(device):
    """Load YOLO11x detector and warm up."""
    assert YOLO_WEIGHTS.exists(), f"YOLO weights not found: {YOLO_WEIGHTS}"
    model = YOLO(str(YOLO_WEIGHTS))
    # Warmup: run a dummy image to trigger CUDA kernel compilation
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model(dummy, imgsz=DETECTOR_IMGSZ, conf=DETECT_CONF, device=device, verbose=False)
    return model


def load_classifier(device):
    """Load DINOv2 embedder from fine-tuned checkpoint, in half precision."""
    assert DINOV2_WEIGHTS.exists(), f"DINOv2 weights not found: {DINOV2_WEIGHTS}"
    assert CLASSIFIER_CHECKPOINT.exists(), f"Classifier checkpoint not found: {CLASSIFIER_CHECKPOINT}"

    model = GroceryEmbedder(
        weights_path=str(DINOV2_WEIGHTS),
        freeze_backbone=True,
    )
    checkpoint = torch.load(CLASSIFIER_CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).half()  # FP16 — no accuracy loss for inference
    model.eval()
    return model


def load_ref_embeddings(device):
    """Load pre-computed reference embeddings."""
    assert REF_EMBEDDINGS.exists(), f"Ref embeddings not found: {REF_EMBEDDINGS}"
    data = torch.load(REF_EMBEDDINGS, map_location=device, weights_only=True)
    ref_embs = data["embeddings"].to(device).half()  # [C, 768] L2-normalized, FP16
    ref_ids = data["category_ids"]  # [C] int64
    return ref_embs, ref_ids


def extract_and_transform_crops(image, boxes):
    """Crop detected regions and apply eval transform, return stacked tensor.

    Args:
        image: PIL Image (RGB)
        boxes: tensor [N, 4] in xyxy format (pixels)

    Returns:
        tensor [M, 3, 518, 518] (FP16), valid_indices list
    """
    w, h = image.size
    tensors = []
    valid_indices = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image.crop((x1, y1, x2, y2))
        t = to_tensor(crop)
        t = resize(t, [CLASSIFIER_RESIZE], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        t = center_crop(t, [CLASSIFIER_SIZE, CLASSIFIER_SIZE])
        t = normalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        tensors.append(t)
        valid_indices.append(i)

    if not tensors:
        return None, []

    return torch.stack(tensors).half(), valid_indices


@torch.no_grad()
def classify_batch(crop_tensors, model, ref_embs, ref_ids, device):
    """Classify pre-transformed crop tensors via cosine similarity.

    Args:
        crop_tensors: [N, 3, 518, 518] FP16 tensor
        model: DINOv2 embedder (FP16)

    Returns:
        category_ids: list[int], cos_scores: list[float]
    """
    category_ids = []
    cos_scores = []
    n = crop_tensors.shape[0]

    for i in range(0, n, CLASSIFY_BATCH):
        batch = crop_tensors[i:i + CLASSIFY_BATCH].to(device)
        embs = model(batch)  # [B, 768] FP16
        embs = F.normalize(embs.float(), dim=1).half()

        sim = embs @ ref_embs.T  # [B, C]
        best_scores, best_indices = sim.max(dim=1)

        for j in range(batch.shape[0]):
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
    detector = load_detector(device)
    classifier = load_classifier(device)
    ref_embs, ref_ids = load_ref_embeddings(device)
    print(f"Models loaded + warmed up in {time.time() - t0:.1f}s")
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
        img_np = np.array(image)

        # Stage 1: Detect (pass numpy array — avoids re-reading from disk)
        results = detector(
            img_np,
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

        # Stage 2: Crop, transform, classify
        crop_tensors, valid_indices = extract_and_transform_crops(image, xyxy)
        if crop_tensors is None:
            continue

        cat_ids, cos_scores = classify_batch(
            crop_tensors, classifier, ref_embs, ref_ids, device
        )

        # Build COCO predictions
        for j, vi in enumerate(valid_indices):
            x1, y1, x2, y2 = xyxy[vi].tolist()
            predictions.append({
                "image_id": image_id,
                "category_id": cat_ids[j],
                "bbox": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2 - x1, 2),
                    round(y2 - y1, 2),
                ],
                "score": round(float(det_scores[vi]) * cos_scores[j], 4),
            })

        if (img_idx + 1) % 10 == 0 or img_idx == 0:
            elapsed = time.time() - t_start
            rate = (img_idx + 1) / elapsed
            print(f"  [{img_idx + 1}/{len(image_paths)}] "
                  f"{len(boxes)} dets, {elapsed:.1f}s, {rate:.1f} img/s")

    # --- Summary ---
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f)

    total_time = time.time() - t_start
    print(f"\nDone! {len(predictions)} predictions from {len(image_paths)} images")
    print(f"Saved to: {output_path}")
    print(f"Total: {total_time:.1f}s ({total_time / max(len(image_paths), 1):.2f}s/image)")


if __name__ == "__main__":
    main()
