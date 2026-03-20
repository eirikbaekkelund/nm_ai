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

ROOT = Path(__file__).resolve().parent
YOLO_WEIGHTS = ROOT / "models" / "yolo_best.pt"
CLASSIFIER_CHECKPOINT = ROOT / "models" / "classifier_best.pt"
REF_EMBEDDINGS = ROOT / "models" / "ref_embeddings.pt"

DETECT_CONF = 0.25
CLASSIFY_BATCH = 128
UNKNOWN_CATEGORY_ID = 355
UNKNOWN_THRESHOLD = 0.0

_NORM_MEAN = None
_NORM_STD = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_file", type=str, default="predictions.json")
    p.add_argument("--detect_conf", type=float, default=DETECT_CONF)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def load_detector(device):
    model = YOLO(str(YOLO_WEIGHTS))
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model(dummy, imgsz=DETECTOR_IMGSZ, conf=DETECT_CONF, device=device,
          half=True, verbose=False)
    return model


def load_classifier(device):
    model = GroceryEmbedder(weights_path=None, freeze_backbone=True)
    checkpoint = torch.load(CLASSIFIER_CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).half().eval()


def load_ref_embeddings(device):
    data = torch.load(REF_EMBEDDINGS, map_location=device, weights_only=True)
    return data["embeddings"].to(device).half(), data["category_ids"]


def extract_and_transform_crops(img_tensor, boxes):
    _, h, w = img_tensor.shape
    crops = []
    valid_indices = []

    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i].int().tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = img_tensor[:, y1:y2, x1:x2]
        crop = resize(crop, [CLASSIFIER_RESIZE],
                      interpolation=InterpolationMode.BICUBIC, antialias=True)
        crop = center_crop(crop, [CLASSIFIER_SIZE, CLASSIFIER_SIZE])
        crops.append(crop)
        valid_indices.append(i)

    if not crops:
        return None, []

    batch = torch.stack(crops)
    batch = (batch - _NORM_MEAN) / _NORM_STD
    return batch.half(), valid_indices


@torch.no_grad()
def classify_batch(crop_tensors, model, ref_embs, ref_ids):
    category_ids = []
    cos_scores = []
    n = crop_tensors.shape[0]

    for i in range(0, n, CLASSIFY_BATCH):
        batch = crop_tensors[i:i + CLASSIFY_BATCH]
        embs = model(batch)
        embs = F.normalize(embs.float(), dim=1).half()

        sim = embs @ ref_embs.T
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
    stem = Path(filename).stem
    digits = "".join(c for c in stem if c.isdigit())
    return int(digits) if digits else hash(stem) % 100000


def main():
    global _NORM_MEAN, _NORM_STD
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    _NORM_MEAN = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    _NORM_STD = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    detector = load_detector(device)
    classifier = load_classifier(device)
    ref_embs, ref_ids = load_ref_embeddings(device)

    input_dir = Path(args.input_dir)
    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    )

    predictions = []

    for img_path in image_paths:
        img_np = np.array(Image.open(img_path).convert("RGB"))
        image_id = image_id_from_filename(img_path.name)

        results = detector(
            img_np, imgsz=DETECTOR_IMGSZ, conf=args.detect_conf,
            device=device, half=True, verbose=False,
        )
        boxes = results[0].boxes

        if len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.cpu()
        det_scores = boxes.conf.cpu()

        img_tensor = (
            torch.from_numpy(img_np)
            .permute(2, 0, 1)
            .to(device=device, dtype=torch.float32)
            .div_(255.0)
        )
        crop_tensors, valid_indices = extract_and_transform_crops(img_tensor, xyxy)
        del img_tensor

        if crop_tensors is None:
            continue

        cat_ids, cos_scores = classify_batch(
            crop_tensors, classifier, ref_embs, ref_ids
        )
        del crop_tensors

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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f)


if __name__ == "__main__":
    main()
