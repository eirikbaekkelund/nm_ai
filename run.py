"""Competition submission — YOLO ONNX detector + native DINOv2 classifier.

Detects grocery products on shelf images (YOLO ONNX) and classifies
them (DINOv2 via timm + cosine similarity to reference embeddings).

Imports: argparse, json, pathlib, numpy, PIL, torch, torchvision, onnxruntime, timm.
All sandbox-safe — no banned imports.

Usage (sandbox):
    python run.py --input /data/images --output /output/predictions.json

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
from torchvision.ops import nms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import center_crop, resize
import onnxruntime as ort

from embedder import GroceryEmbedder

# ── Constants ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
YOLO_WEIGHTS = ROOT / "models" / "yolo.onnx"
CLASSIFIER_WEIGHTS = ROOT / "models" / "classifier.pt"
REF_EMBEDDINGS = ROOT / "models" / "ref_embeddings.pt"

DETECTOR_IMGSZ = 1280
CLASSIFIER_RESIZE = 546
CLASSIFIER_SIZE = 518
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DETECT_CONF = 0.25
NMS_IOU = 0.7
CLASSIFY_BATCH = 128
UNKNOWN_CATEGORY_ID = 355
UNKNOWN_THRESHOLD = 0.0

_NORM_MEAN = None
_NORM_STD = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, dest="input_dir")
    p.add_argument("--output", type=str, default="predictions.json", dest="output_file")
    p.add_argument("--detect_conf", type=float, default=DETECT_CONF)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


# ── YOLO preprocessing ──────────────────────────────────────────────────────


def letterbox(img_tensor, target_size=1280):
    """Letterbox [3, H, W] float32 tensor to [3, target_size, target_size].

    Returns: (padded_tensor, scale, pad_x, pad_y)
    """
    _, h, w = img_tensor.shape
    scale = min(target_size / h, target_size / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    resized = F.interpolate(
        img_tensor.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    pad_y = (target_size - new_h) // 2
    pad_x = (target_size - new_w) // 2
    pad_bottom = target_size - new_h - pad_y
    pad_right = target_size - new_w - pad_x

    padded = F.pad(resized, (pad_x, pad_right, pad_y, pad_bottom), value=114 / 255)
    return padded, scale, pad_x, pad_y


# ── YOLO postprocessing ─────────────────────────────────────────────────────


def yolo11_postprocess(output, conf_thresh, scale, pad_x, pad_y, orig_h, orig_w):
    """Decode YOLO11x output [1, 5, N] — (cx, cy, w, h, conf) single-class."""
    pred = output[0].float().T  # [N, 5]

    conf = pred[:, 4]
    mask = conf > conf_thresh
    pred = pred[mask]

    if pred.shape[0] == 0:
        return (
            torch.empty((0, 4), device=output.device),
            torch.empty((0,), device=output.device),
        )

    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    scores = pred[:, 4]

    keep = nms(boxes, scores, NMS_IOU)
    boxes = boxes[keep]
    scores = scores[keep]

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_h)

    return boxes, scores


def yolo26_postprocess(output, conf_thresh, scale, pad_x, pad_y, orig_h, orig_w):
    """Decode YOLO26 output [1, 300, 6] — (x1, y1, x2, y2, conf, class) NMS-free."""
    pred = output[0].float()  # [300, 6]

    scores = pred[:, 4]
    mask = scores > conf_thresh
    pred = pred[mask]

    if pred.shape[0] == 0:
        return (
            torch.empty((0, 4), device=output.device),
            torch.empty((0,), device=output.device),
        )

    boxes = pred[:, :4]
    scores = pred[:, 4]

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_h)

    return boxes, scores


def yolo_postprocess(output, conf_thresh, scale, pad_x, pad_y, orig_h, orig_w):
    """Auto-detect YOLO version from output shape and dispatch."""
    shape = output.shape
    if len(shape) == 3 and shape[1] == 5:
        # YOLO11x: [1, 5, N] — needs NMS
        return yolo11_postprocess(output, conf_thresh, scale, pad_x, pad_y, orig_h, orig_w)
    elif len(shape) == 3 and shape[2] == 6:
        # YOLO26: [1, 300, 6] — NMS-free
        return yolo26_postprocess(output, conf_thresh, scale, pad_x, pad_y, orig_h, orig_w)
    else:
        raise ValueError(f"Unknown YOLO output shape: {shape}")


# ── Classifier pipeline ─────────────────────────────────────────────────────


def extract_and_transform_crops(img_tensor, boxes):
    """Crop, resize, center-crop, normalize for classifier."""
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
    batch = (batch - _NORM_MEAN) / _NORM_STD
    return batch, valid_indices


@torch.no_grad()
def classify_crops(crop_tensors, model, ref_embs, ref_ids, device):
    """Classify crops by cosine similarity to reference embeddings."""
    category_ids = []
    cos_scores = []
    n = crop_tensors.shape[0]

    for i in range(0, n, CLASSIFY_BATCH):
        batch = crop_tensors[i : i + CLASSIFY_BATCH].to(device)

        embs = model(batch)
        embs = F.normalize(embs.float(), dim=1)
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


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    global _NORM_MEAN, _NORM_STD
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    _NORM_MEAN = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    _NORM_STD = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    # Load YOLO ONNX
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    yolo_sess = ort.InferenceSession(str(YOLO_WEIGHTS), providers=providers)
    yolo_input_name = yolo_sess.get_inputs()[0].name

    # Probe YOLO output shape to determine input dtype
    probe_input = np.zeros((1, 3, DETECTOR_IMGSZ, DETECTOR_IMGSZ), dtype=np.float16)
    try:
        probe_out = yolo_sess.run(None, {yolo_input_name: probe_input})[0]
        yolo_input_dtype = np.float16
    except Exception:
        # YOLO26 ONNX may require FP32 input
        probe_input = np.zeros((1, 3, DETECTOR_IMGSZ, DETECTOR_IMGSZ), dtype=np.float32)
        probe_out = yolo_sess.run(None, {yolo_input_name: probe_input})[0]
        yolo_input_dtype = np.float32
    yolo_version = "YOLO26" if probe_out.shape[-1] == 6 else "YOLO11"
    print(f"YOLO: {yolo_version}, input dtype: {yolo_input_dtype}, output shape: {probe_out.shape}")
    del probe_out

    # Load native DINOv2 classifier
    cls_model = GroceryEmbedder()
    sd = torch.load(str(CLASSIFIER_WEIGHTS), map_location="cpu", weights_only=True)
    cls_model.load_state_dict(sd)
    cls_model = cls_model.half().to(device).eval()

    # Load reference embeddings
    ref_data = torch.load(str(REF_EMBEDDINGS), map_location=device, weights_only=True)
    ref_embs = F.normalize(ref_data["embeddings"].to(device).float(), dim=1)
    ref_ids = ref_data["category_ids"]

    input_dir = Path(args.input_dir)
    image_paths = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    )

    predictions = []

    for img_path in image_paths:
        img_np = np.array(Image.open(img_path).convert("RGB"))
        image_id = image_id_from_filename(img_path.name)
        orig_h, orig_w = img_np.shape[:2]

        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(device=device, dtype=torch.float32).div_(255.0)

        # YOLO: letterbox → numpy → ORT detect → torch postprocess
        lb_tensor, scale, pad_x, pad_y = letterbox(img_tensor, DETECTOR_IMGSZ)
        if yolo_input_dtype == np.float16:
            lb_np = lb_tensor.unsqueeze(0).half().cpu().numpy()
        else:
            lb_np = lb_tensor.unsqueeze(0).cpu().numpy()

        yolo_out_np = yolo_sess.run(None, {yolo_input_name: lb_np})[0]
        yolo_out = torch.from_numpy(yolo_out_np).to(device)

        boxes_xyxy, det_scores = yolo_postprocess(
            yolo_out,
            args.detect_conf,
            scale,
            pad_x,
            pad_y,
            orig_h,
            orig_w,
        )
        del lb_tensor, lb_np, yolo_out_np, yolo_out

        if boxes_xyxy.shape[0] == 0:
            del img_tensor
            continue

        # Classify crops
        crop_tensors, valid_indices = extract_and_transform_crops(
            img_tensor,
            boxes_xyxy,
        )
        del img_tensor

        if crop_tensors is None:
            continue

        # Convert crops to FP16 to match model dtype
        crop_tensors = crop_tensors.half()

        cat_ids, cos_scores = classify_crops(
            crop_tensors,
            cls_model,
            ref_embs,
            ref_ids,
            device,
        )
        del crop_tensors

        for j, vi in enumerate(valid_indices):
            x1, y1, x2, y2 = boxes_xyxy[vi].tolist()
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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
