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
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import center_crop, resize
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
CLASSIFY_BATCH = 128
UNKNOWN_CATEGORY_ID = 355
UNKNOWN_THRESHOLD = 0.0  # cosine sim below this -> unknown (disabled)

# Pre-computed normalization tensors on GPU (set in main)
_NORM_MEAN = None
_NORM_STD = None


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
    """Load YOLO11x detector, warm up in FP16."""
    assert YOLO_WEIGHTS.exists(), f"YOLO weights not found: {YOLO_WEIGHTS}"
    model = YOLO(str(YOLO_WEIGHTS))
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model(dummy, imgsz=DETECTOR_IMGSZ, conf=DETECT_CONF, device=device,
          half=True, verbose=False)
    return model


def load_classifier(device):
    """Load DINOv2 embedder from fine-tuned checkpoint, FP16."""
    assert CLASSIFIER_CHECKPOINT.exists(), f"Checkpoint not found: {CLASSIFIER_CHECKPOINT}"

    # weights_path=None: skip loading base DINOv2 weights since
    # classifier_best.pt already contains the full fine-tuned backbone
    model = GroceryEmbedder(
        weights_path=None,
        freeze_backbone=True,
    )
    checkpoint = torch.load(CLASSIFIER_CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).half().eval()
    return model


def load_ref_embeddings(device):
    """Load pre-computed reference embeddings."""
    assert REF_EMBEDDINGS.exists(), f"Ref embeddings not found: {REF_EMBEDDINGS}"
    data = torch.load(REF_EMBEDDINGS, map_location=device, weights_only=True)
    ref_embs = data["embeddings"].to(device).half()  # [C, 768]
    ref_ids = data["category_ids"]  # [C]
    return ref_embs, ref_ids


def extract_and_transform_crops(img_tensor, boxes):
    """Crop from GPU tensor, transform entirely on GPU.

    Args:
        img_tensor: [3, H, W] float32 GPU tensor, range [0, 1]
        boxes: [N, 4] xyxy tensor (CPU)

    Returns:
        [M, 3, 518, 518] FP16 GPU tensor, valid_indices list
    """
    _, h, w = img_tensor.shape
    crops = []
    valid_indices = []

    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i].int().tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = img_tensor[:, y1:y2, x1:x2]  # GPU slice, zero-copy
        crop = resize(crop, [CLASSIFIER_RESIZE],
                      interpolation=InterpolationMode.BICUBIC, antialias=True)
        crop = center_crop(crop, [CLASSIFIER_SIZE, CLASSIFIER_SIZE])
        crops.append(crop)
        valid_indices.append(i)

    if not crops:
        return None, []

    batch = torch.stack(crops)  # [N, 3, 518, 518] already on GPU
    batch = (batch - _NORM_MEAN) / _NORM_STD  # vectorized normalize
    return batch.half(), valid_indices


@torch.no_grad()
def classify_batch(crop_tensors, model, ref_embs, ref_ids):
    """Classify crop tensors via cosine similarity to reference embeddings."""
    category_ids = []
    cos_scores = []
    n = crop_tensors.shape[0]

    for i in range(0, n, CLASSIFY_BATCH):
        batch = crop_tensors[i:i + CLASSIFY_BATCH]
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
    """Extract numeric image_id from filename like 'img_00042.jpg' -> 42."""
    stem = Path(filename).stem
    digits = "".join(c for c in stem if c.isdigit())
    return int(digits) if digits else hash(stem) % 100000


def main():
    global _NORM_MEAN, _NORM_STD
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True

    # Pre-compute normalization tensors on GPU
    _NORM_MEAN = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    _NORM_STD = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

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
    t_detect = 0.0
    t_crop = 0.0
    t_classify = 0.0

    for img_idx, img_path in enumerate(image_paths):
        img_np = np.array(Image.open(img_path).convert("RGB"))
        image_id = image_id_from_filename(img_path.name)

        # Stage 1: Detect (FP16)
        td0 = time.time()
        results = detector(
            img_np,
            imgsz=DETECTOR_IMGSZ,
            conf=args.detect_conf,
            device=device,
            half=True,
            verbose=False,
        )
        boxes = results[0].boxes
        t_detect += time.time() - td0

        if len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.cpu()  # [N, 4]
        det_scores = boxes.conf.cpu()  # [N]

        # Stage 2: GPU crop + transform
        tc0 = time.time()
        img_tensor = (
            torch.from_numpy(img_np)
            .permute(2, 0, 1)
            .to(device=device, dtype=torch.float32)
            .div_(255.0)
        )
        crop_tensors, valid_indices = extract_and_transform_crops(img_tensor, xyxy)
        del img_tensor  # free GPU memory
        t_crop += time.time() - tc0

        if crop_tensors is None:
            continue

        # Stage 3: Classify
        tcl0 = time.time()
        cat_ids, cos_scores = classify_batch(
            crop_tensors, classifier, ref_embs, ref_ids
        )
        del crop_tensors  # free GPU memory
        t_classify += time.time() - tcl0

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
    total_time = time.time() - t_start

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    print(f"\nTiming breakdown:")
    print(f"  Detection:      {t_detect:6.1f}s ({t_detect/max(total_time,1)*100:4.0f}%)")
    print(f"  Crop+transform: {t_crop:6.1f}s ({t_crop/max(total_time,1)*100:4.0f}%)")
    print(f"  Classification: {t_classify:6.1f}s ({t_classify/max(total_time,1)*100:4.0f}%)")
    other = total_time - t_detect - t_crop - t_classify
    print(f"  I/O + overhead:  {other:5.1f}s ({other/max(total_time,1)*100:4.0f}%)")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f)

    print(f"\nDone! {len(predictions)} predictions from {len(image_paths)} images")
    print(f"Saved to: {output_path}")
    print(f"Total: {total_time:.1f}s ({total_time / max(len(image_paths), 1):.2f}s/image)")


if __name__ == "__main__":
    main()
