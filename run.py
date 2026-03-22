"""Competition submission — YOLO ONNX detector + native DINOv2 classifier.

Detects grocery products on shelf images (YOLO ONNX) and classifies
them (DINOv2 via timm + cosine similarity to reference embeddings).

SAHI tiled inference for small object recall on large images.

Imports: argparse, json, pathlib, numpy, PIL, torch, torchvision, onnxruntime,
         timm, ensemble_boxes.
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
from ensemble_boxes import weighted_boxes_fusion

from embedder import GroceryEmbedder

# ── Constants ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
YOLO_WEIGHTS = ROOT / "yolo.onnx"
CLASSIFIER_WEIGHTS = ROOT / "classifier.pt"
REF_EMBEDDINGS = ROOT / "ref_embeddings.pt"

DETECTOR_IMGSZ = 1280
CLASSIFIER_RESIZE = 546
CLASSIFIER_SIZE = 518
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DETECT_CONF = 0.05
NMS_IOU = 0.7
CLASSIFY_BATCH = 128
CROP_BUFFER = 0.05
UNKNOWN_CATEGORY_ID = 355
UNKNOWN_THRESHOLD = 0.0

# CAQE — Context-Aware Query Expansion
CAQE_K = 3
CAQE_ALPHA = 0.5

# SAHI — Slicing Aided Hyper Inference
TILE_SIZE = 640
TILE_OVERLAP = 0.25
MIN_DIM_FOR_TILING = 2000
WBF_IOU_THR = 0.55

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
        return yolo11_postprocess(output, conf_thresh, scale, pad_x, pad_y, orig_h, orig_w)
    elif len(shape) == 3 and shape[2] == 6:
        return yolo26_postprocess(output, conf_thresh, scale, pad_x, pad_y, orig_h, orig_w)
    else:
        raise ValueError(f"Unknown YOLO output shape: {shape}")


# ── SAHI tiled detection ───────────────────────────────────────────────────


def _run_yolo(yolo_sess, yolo_input_name, img_tensor, conf, device, orig_h, orig_w):
    """Letterbox + YOLO inference + postprocess on a single image/tile."""
    lb_tensor, scale, pad_x, pad_y = letterbox(img_tensor, DETECTOR_IMGSZ)
    lb_np = lb_tensor.unsqueeze(0).cpu().numpy()
    yolo_out_np = yolo_sess.run(None, {yolo_input_name: lb_np})[0]
    yolo_out = torch.from_numpy(yolo_out_np).to(device)
    boxes, scores = yolo_postprocess(
        yolo_out, conf, scale, pad_x, pad_y, orig_h, orig_w
    )
    del lb_tensor, lb_np, yolo_out_np, yolo_out
    return boxes, scores


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


def detect_with_tiles(yolo_sess, yolo_input_name, img_tensor, device, conf=DETECT_CONF):
    """Full-image + tiled SAHI detection, merged via WBF.

    Returns (boxes_xyxy [N,4], scores [N]) in original image coordinates.
    """
    _, orig_h, orig_w = img_tensor.shape

    # Pass 1: full-image at 1280
    full_boxes, full_scores = _run_yolo(
        yolo_sess, yolo_input_name, img_tensor, conf, device, orig_h, orig_w
    )

    # Skip tiling for small images
    if max(orig_h, orig_w) <= MIN_DIM_FOR_TILING:
        return full_boxes, full_scores

    # Pass 2: tiled detection
    tiles = generate_tiles(orig_h, orig_w, TILE_SIZE, TILE_OVERLAP)
    all_boxes = [full_boxes.cpu().numpy() if full_boxes.shape[0] > 0
                 else np.empty((0, 4), dtype=np.float32)]
    all_scores = [full_scores.cpu().numpy() if full_scores.shape[0] > 0
                  else np.empty((0,), dtype=np.float32)]

    for tx1, ty1, tx2, ty2 in tiles:
        tile = img_tensor[:, ty1:ty2, tx1:tx2]
        _, tile_h, tile_w = tile.shape
        tboxes, tscores = _run_yolo(
            yolo_sess, yolo_input_name, tile, conf, device, tile_h, tile_w
        )
        if tboxes.shape[0] > 0:
            # Remap tile coords to original image coords
            tboxes[:, [0, 2]] += tx1
            tboxes[:, [1, 3]] += ty1
            tboxes[:, [0, 2]] = tboxes[:, [0, 2]].clamp(0, orig_w)
            tboxes[:, [1, 3]] = tboxes[:, [1, 3]].clamp(0, orig_h)
            all_boxes.append(tboxes.cpu().numpy())
            all_scores.append(tscores.cpu().numpy())
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
        return (torch.empty((0, 4), device=device),
                torch.empty((0,), device=device))

    fb, fs, _ = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        iou_thr=WBF_IOU_THR, skip_box_thr=0.0,
    )
    fb[:, [0, 2]] *= orig_w
    fb[:, [1, 3]] *= orig_h
    return (torch.from_numpy(fb).float().to(device),
            torch.from_numpy(fs).float().to(device))


# ── Classifier pipeline ─────────────────────────────────────────────────────


def extract_and_transform_crops(img_tensor, boxes):
    """Crop with 5% padding, resize, center-crop, normalize for classifier."""
    _, h, w = img_tensor.shape
    crops = []
    valid_indices = []

    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i].tolist()
        bw, bh = x2 - x1, y2 - y1
        pad_x = bw * CROP_BUFFER
        pad_y = bh * CROP_BUFFER
        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(w, int(x2 + pad_x))
        y2 = min(h, int(y2 + pad_y))
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
def embed_crops(crop_tensors, model, device):
    """Embed all crops → [N, D] L2-normalized embeddings."""
    all_embs = []
    n = crop_tensors.shape[0]
    for i in range(0, n, CLASSIFY_BATCH):
        batch = crop_tensors[i : i + CLASSIFY_BATCH].to(device)
        embs = model(batch)
        embs = F.normalize(embs.float(), dim=1)
        all_embs.append(embs)
    return torch.cat(all_embs, dim=0)


def apply_caqe(embeddings, boxes_xyxy, k=CAQE_K, alpha=CAQE_ALPHA):
    """Context-Aware Query Expansion — boost each embedding with spatial neighbors."""
    n = embeddings.shape[0]
    if k <= 0 or n <= 1 or alpha >= 1.0:
        return embeddings
    eff_k = min(k, n - 1)
    centroids = torch.stack([
        (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2,
        (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2,
    ], dim=1)
    dists = torch.cdist(centroids.unsqueeze(0).float(), centroids.unsqueeze(0).float()).squeeze(0)
    dists.fill_diagonal_(float("inf"))
    _, nn_indices = dists.topk(eff_k, dim=1, largest=False)
    neighbor_mean = embeddings[nn_indices].mean(dim=1)
    expanded = alpha * embeddings + (1 - alpha) * neighbor_mean
    return F.normalize(expanded, dim=1)


def match_to_refs(embeddings, ref_embs, ref_ids):
    """Match [N, D] embeddings to refs → (category_ids, cos_scores)."""
    sim = embeddings @ ref_embs.T
    best_scores, best_indices = sim.max(dim=1)
    category_ids = []
    cos_scores = []
    for j in range(embeddings.shape[0]):
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

    # Probe YOLO output shape to auto-detect version
    probe_input = np.zeros((1, 3, DETECTOR_IMGSZ, DETECTOR_IMGSZ), dtype=np.float32)
    probe_out = yolo_sess.run(None, {yolo_input_name: probe_input})[0]
    yolo_version = "YOLO26" if probe_out.shape[-1] == 6 else "YOLO11"
    print(f"YOLO: {yolo_version}, output shape: {probe_out.shape}")
    del probe_out

    # Load native DINOv2 classifier
    cls_model = GroceryEmbedder()
    sd = torch.load(str(CLASSIFIER_WEIGHTS), map_location="cpu", weights_only=True)
    cls_model.load_state_dict(sd)
    cls_model = cls_model.to(device).eval()

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

        # SAHI: full-image + tiled detection, merged via WBF
        boxes_xyxy, det_scores = detect_with_tiles(
            yolo_sess, yolo_input_name,
            img_tensor, device, conf=args.detect_conf,
        )

        if boxes_xyxy.shape[0] == 0:
            del img_tensor
            continue

        # Classify crops
        crop_tensors, valid_indices = extract_and_transform_crops(
            img_tensor, boxes_xyxy,
        )
        del img_tensor

        if crop_tensors is None:
            continue

        embeddings = embed_crops(crop_tensors, cls_model, device)
        del crop_tensors

        # CAQE: expand embeddings with spatial context from neighbors
        if CAQE_K > 0 and len(valid_indices) > 1:
            valid_boxes = boxes_xyxy[valid_indices]
            embeddings = apply_caqe(embeddings, valid_boxes, k=CAQE_K, alpha=CAQE_ALPHA)

        cat_ids, cos_scores = match_to_refs(embeddings, ref_embs, ref_ids)
        del embeddings

        for j, vi in enumerate(valid_indices):
            x1, y1, x2, y2 = boxes_xyxy[vi].tolist()
            score_val = float(det_scores[vi]) * cos_scores[j]
            if not np.isfinite(score_val):
                score_val = 0.0001
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
                    "score": round(score_val, 4),
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
