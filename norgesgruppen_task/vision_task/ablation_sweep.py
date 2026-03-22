"""
Inference-time ablation sweep — evaluates detection + classification strategies
on the validation set without retraining.

Axes:
  det_conf      — YOLO confidence threshold
  scales        — single or multi-scale WBF
  knn_k         — k-NN weighted voting for classification
  query_tta     — horizontal flip TTA on query crops
  unknown_threshold — cosine sim below this → unknown_product

Evaluation via pycocotools mAP against COCO ground truth (val split).

Speed optimization: embeddings are pre-computed once per unique (scales, query_tta)
combo at the lowest det_conf. Higher conf thresholds just filter pre-computed results.
knn_k and unknown_threshold only change matching logic — no re-embedding needed.

Usage:
    python -m vision_task.ablation_sweep
    python -m vision_task.ablation_sweep --mode combinatorial --axes det_conf,knn_k
"""

import argparse
import itertools
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from vision_task.caqe import apply_caqe
from vision_task.config import CROP_BUFFER
from vision_task.data.transforms import get_eval_transform
from vision_task.predict import image_id_from_filename, load_classifier

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

UNKNOWN_CATEGORY_ID = 355

# Baseline values (updated from previous sweep winners)
BASELINE = {
    "det_conf": 0.05,
    "scales": [1280],
    "knn_k": 1,
    "query_tta": False,
    "unknown_threshold": 0.0,
    "caqe_k": 0,
    "caqe_alpha": 0.5,
}

# Sweep grid — only axes with unexplored values
# Previously settled: det_conf=0.05, scales=[1280], unknown_threshold=0.0
SWEEP_GRID = {
    "det_conf": [0.05],
    "scales": [[1280]],
    "knn_k": [1, 3, 5],
    "query_tta": [False, True],
    "unknown_threshold": [0.0],
    "caqe_k": [0, 2, 3, 5],
    "caqe_alpha": [0.3, 0.5, 0.7],
}


@dataclass
class SweepConfig:
    det_conf: float = 0.25
    scales: list = field(default_factory=lambda: [1280])
    knn_k: int = 1
    query_tta: bool = False
    unknown_threshold: float = 0.0
    caqe_k: int = 0
    caqe_alpha: float = 0.5

    def label(self) -> str:
        parts = []
        if self.det_conf != BASELINE["det_conf"]:
            parts.append(f"conf={self.det_conf}")
        if self.scales != BASELINE["scales"]:
            parts.append(f"scales={self.scales}")
        if self.knn_k != BASELINE["knn_k"]:
            parts.append(f"knn={self.knn_k}")
        if self.query_tta != BASELINE["query_tta"]:
            parts.append("tta=True")
        if self.unknown_threshold != BASELINE["unknown_threshold"]:
            parts.append(f"unk={self.unknown_threshold}")
        if self.caqe_k != BASELINE["caqe_k"]:
            parts.append(f"caqe_k={self.caqe_k}")
        if self.caqe_alpha != BASELINE["caqe_alpha"]:
            parts.append(f"caqe_a={self.caqe_alpha}")
        return ", ".join(parts) if parts else "baseline"


# ---------------------------------------------------------------------------
# COCO ground-truth helpers
# ---------------------------------------------------------------------------


def get_val_image_ids(annotations_path: Path, val_dir: Path) -> list:
    """Match val filenames to COCO image_ids."""
    with open(annotations_path, "r") as f:
        coco_data = json.load(f)

    name_to_id = {img["file_name"]: img["id"] for img in coco_data["images"]}
    val_ids = []
    for p in sorted(val_dir.iterdir()):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            if p.name in name_to_id:
                val_ids.append(name_to_id[p.name])
    return val_ids


def build_val_coco_gt(annotations_path: Path, val_image_ids: set):
    """Build detection + classification COCO GT objects for val images only."""
    with open(annotations_path, "r") as f:
        coco_data = json.load(f)

    val_images = [img for img in coco_data["images"] if img["id"] in val_image_ids]
    val_anns = [ann for ann in coco_data["annotations"] if ann["image_id"] in val_image_ids]

    # Detection GT: all category_id collapsed to 0
    det_gt = {
        "images": val_images,
        "annotations": [{**ann, "category_id": 0, "id": ann["id"]} for ann in val_anns],
        "categories": [{"id": 0, "name": "product"}],
    }

    # Classification GT: real category_ids
    cls_cats = {ann["category_id"] for ann in val_anns}
    cls_gt = {
        "images": val_images,
        "annotations": val_anns,
        "categories": [cat for cat in coco_data["categories"] if cat["id"] in cls_cats],
    }

    return coco_from_dict(det_gt), coco_from_dict(cls_gt)


def coco_from_dict(data: dict) -> COCO:
    """Create a COCO object from a dict without writing a temp file."""
    coco = COCO()
    coco.dataset = data
    coco.createIndex()
    return coco


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class DetectionCache:
    """Cache YOLO detections per (image_path, imgsz) at conf=0.01.

    All conf thresholds filter from cached results — no re-running YOLO.
    """

    def __init__(self, yolo_model, device):
        self._yolo = yolo_model
        self._device = device
        self._cache = {}  # (str(image_path), imgsz) -> (boxes_xyxy, scores)

    def get(self, image_path: Path, imgsz: int) -> tuple:
        """Returns (boxes_xyxy [N,4] np, scores [N] np) at conf=0.01."""
        key = (str(image_path), imgsz)
        if key not in self._cache:
            results = self._yolo.predict(
                source=str(image_path),
                conf=0.01,
                imgsz=imgsz,
                device=self._device,
                verbose=False,
            )
            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()
            else:
                boxes = np.empty((0, 4), dtype=np.float32)
                scores = np.empty((0,), dtype=np.float32)
            self._cache[key] = (boxes, scores)
        return self._cache[key]


def detect_single_scale(cache: DetectionCache, image_path: Path, imgsz: int, conf: float):
    """Filter cached detections by confidence threshold."""
    boxes, scores = cache.get(image_path, imgsz)
    mask = scores >= conf
    return boxes[mask], scores[mask]


def detect_multiscale_wbf(
    cache: DetectionCache, image_path: Path, scales: list, conf: float, img_w: int, img_h: int, iou_thr: float = 0.6
):
    """Multi-scale detection with Weighted Box Fusion."""
    from ensemble_boxes import weighted_boxes_fusion

    boxes_list = []
    scores_list = []
    labels_list = []

    for scale in scales:
        boxes, scores = detect_single_scale(cache, image_path, scale, conf)
        if len(boxes) == 0:
            boxes_list.append(np.empty((0, 4), dtype=np.float32))
            scores_list.append(np.empty((0,), dtype=np.float32))
            labels_list.append(np.empty((0,), dtype=np.float32))
            continue
        # Normalize to [0, 1]
        norm_boxes = boxes.copy()
        norm_boxes[:, [0, 2]] /= img_w
        norm_boxes[:, [1, 3]] /= img_h
        # Clip to [0, 1]
        norm_boxes = np.clip(norm_boxes, 0, 1)
        boxes_list.append(norm_boxes)
        scores_list.append(scores)
        labels_list.append(np.zeros(len(scores), dtype=np.float32))

    if all(len(b) == 0 for b in boxes_list):
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    fused_boxes, fused_scores, _ = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        iou_thr=iou_thr,
        skip_box_thr=0.0,
    )

    # Denormalize
    fused_boxes[:, [0, 2]] *= img_w
    fused_boxes[:, [1, 3]] *= img_h

    return fused_boxes, fused_scores


# ---------------------------------------------------------------------------
# Crop extraction
# ---------------------------------------------------------------------------


def extract_crops(img: Image.Image, boxes_xyxy: np.ndarray) -> tuple:
    """Crop PIL images with CROP_BUFFER padding. Returns (crops, valid_indices)."""
    w, h = img.size
    crops = []
    valid = []
    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = boxes_xyxy[i]
        bw, bh = x2 - x1, y2 - y1
        pad_x = bw * CROP_BUFFER
        pad_y = bh * CROP_BUFFER
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(w, int(x2 + pad_x))
        cy2 = min(h, int(y2 + pad_y))
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        crops.append(img.crop((cx1, cy1, cx2, cy2)))
        valid.append(i)
    return crops, valid


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@torch.no_grad()
def embed_crops_standard(crops: list, model, device, batch_size: int = 128) -> torch.Tensor:
    """Standard eval transform → model → L2-normalize. Returns [N, 768]."""
    transform = get_eval_transform()
    all_embs = []
    for i in range(0, len(crops), batch_size):
        batch_pil = crops[i : i + batch_size]
        batch = torch.stack([transform(img) for img in batch_pil]).to(device)
        embs = model(batch)
        embs = F.normalize(embs.float(), dim=1)
        all_embs.append(embs)
    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def embed_crops_with_tta(crops: list, model, device, batch_size: int = 128) -> torch.Tensor:
    """Original + hflip → average → L2-normalize. Returns [N, 768]."""
    import torchvision.transforms.functional as TF

    transform = get_eval_transform()
    all_embs = []
    for i in range(0, len(crops), batch_size):
        batch_pil = crops[i : i + batch_size]
        # Original
        orig = torch.stack([transform(img) for img in batch_pil]).to(device)
        embs_orig = model(orig).float()
        # Horizontal flip
        flipped = torch.stack([transform(TF.hflip(img)) for img in batch_pil]).to(device)
        embs_flip = model(flipped).float()
        # Average and normalize
        embs = F.normalize((embs_orig + embs_flip) / 2, dim=1)
        all_embs.append(embs)
    return torch.cat(all_embs, dim=0)


def classify_knn(
    query_embs: torch.Tensor, ref_embs: torch.Tensor, ref_ids: torch.Tensor, k: int, unknown_threshold: float
) -> tuple:
    """k-NN weighted voting. Returns (category_ids [N], cos_scores [N])."""
    sim = query_embs @ ref_embs.T  # [N_query, N_ref]

    if k == 1:
        best_scores, best_indices = sim.max(dim=1)
        cat_ids = []
        scores = []
        for j in range(len(query_embs)):
            score = best_scores[j].item()
            if score < unknown_threshold:
                cat_ids.append(UNKNOWN_CATEGORY_ID)
            else:
                cat_ids.append(ref_ids[best_indices[j]].item())
            scores.append(score)
        return cat_ids, scores

    # k-NN: top-k neighbors, similarity-weighted vote
    topk_scores, topk_indices = sim.topk(k, dim=1)  # [N, k]
    cat_ids = []
    scores = []
    for j in range(len(query_embs)):
        # Gather category_ids for top-k neighbors
        neighbor_cats = ref_ids[topk_indices[j]]  # [k]
        neighbor_sims = topk_scores[j]  # [k]

        # Weighted vote: sum similarities per category
        vote = {}
        for c_idx in range(k):
            cat = neighbor_cats[c_idx].item()
            vote[cat] = vote.get(cat, 0.0) + neighbor_sims[c_idx].item()

        best_cat = max(vote, key=vote.get)
        best_score = vote[best_cat] / k  # normalize by k

        if best_score < unknown_threshold:
            cat_ids.append(UNKNOWN_CATEGORY_ID)
        else:
            cat_ids.append(best_cat)
        scores.append(best_score)

    return cat_ids, scores


# ---------------------------------------------------------------------------
# COCO evaluation
# ---------------------------------------------------------------------------


def evaluate_coco_map(predictions: list, det_gt: COCO, cls_gt: COCO) -> dict:
    """Compute detection + classification mAP@50 via pycocotools.

    Returns dict with det_mAP50, cls_mAP50, combined (0.7*det + 0.3*cls).
    """
    metrics = {"det_mAP50": 0.0, "cls_mAP50": 0.0, "combined": 0.0, "n_preds": len(predictions)}

    if not predictions:
        return metrics

    # Detection eval: all category_id → 0
    det_preds = [{**p, "category_id": 0} for p in predictions]
    det_map = _run_coco_eval(det_preds, det_gt)
    metrics["det_mAP50"] = det_map

    # Classification eval: real category_ids
    cls_map = _run_coco_eval(predictions, cls_gt)
    metrics["cls_mAP50"] = cls_map

    metrics["combined"] = 0.7 * det_map + 0.3 * cls_map
    return metrics


def _run_coco_eval(predictions: list, coco_gt: COCO) -> float:
    """Run COCOeval and return mAP@50."""
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.iouThrs = np.array([0.5])
    coco_eval.params.maxDets = [1, 10, 300]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # stats[0] uses hardcoded maxDets=100 (returns -1 since we use [1,10,300])
    # stats[1] = AP @ IoU=0.50, area=all, maxDets=params.maxDets[2]=300
    return float(coco_eval.stats[1])


# ---------------------------------------------------------------------------
# Pre-computed embedding store (main speed optimization)
# ---------------------------------------------------------------------------


class EmbeddingStore:
    """Pre-compute and cache embeddings per unique (scales, query_tta) combo.

    For each combo, detections are run at the minimum conf across all configs.
    Embeddings are computed once. Higher conf thresholds filter the cached results.

    This reduces ~14 DINOv2 forward passes to just 2-3 (one per unique combo).
    """

    def __init__(
        self, det_cache: DetectionCache, cls_model, ref_embs, ref_ids, device, image_paths: list, batch_size: int = 128
    ):
        self._det_cache = det_cache
        self._cls_model = cls_model
        self._ref_embs = ref_embs
        self._ref_ids = ref_ids
        self._device = device
        self._image_paths = image_paths
        self._batch_size = batch_size
        # Cache: (scales_key, query_tta) -> list of per-image dicts
        self._cache = {}

    def _scales_key(self, scales: list) -> tuple:
        return tuple(scales)

    def precompute(self, scales: list, query_tta: bool, min_conf: float):
        """Pre-compute embeddings for all val images at min_conf."""
        key = (self._scales_key(scales), query_tta)
        if key in self._cache:
            return

        logger.info("Pre-computing embeddings: scales=%s, tta=%s, min_conf=%.2f", scales, query_tta, min_conf)
        t0 = time.time()

        per_image_data = []
        total_crops = 0

        for img_path in self._image_paths:
            img = Image.open(img_path).convert("RGB")
            img_w, img_h = img.size
            image_id = image_id_from_filename(img_path.name)

            # Detect at min_conf (superset of all higher conf)
            if len(scales) == 1:
                boxes, det_scores = detect_single_scale(self._det_cache, img_path, scales[0], min_conf)
            else:
                boxes, det_scores = detect_multiscale_wbf(self._det_cache, img_path, scales, min_conf, img_w, img_h)

            if len(boxes) == 0:
                per_image_data.append(
                    {
                        "image_id": image_id,
                        "boxes": np.empty((0, 4), dtype=np.float32),
                        "det_scores": np.empty((0,), dtype=np.float32),
                        "valid_indices": [],
                        "embeddings": None,
                    }
                )
                continue

            # Crop
            crops, valid_indices = extract_crops(img, boxes)
            if not crops:
                per_image_data.append(
                    {
                        "image_id": image_id,
                        "boxes": boxes,
                        "det_scores": det_scores,
                        "valid_indices": [],
                        "embeddings": None,
                    }
                )
                continue

            # Embed
            if query_tta:
                embs = embed_crops_with_tta(crops, self._cls_model, self._device, self._batch_size)
            else:
                embs = embed_crops_standard(crops, self._cls_model, self._device, self._batch_size)

            total_crops += len(crops)
            per_image_data.append(
                {
                    "image_id": image_id,
                    "boxes": boxes,
                    "det_scores": det_scores,
                    "valid_indices": valid_indices,
                    "embeddings": embs,
                }
            )

        self._cache[key] = per_image_data
        logger.info("  Embedded %d crops in %.1fs", total_crops, time.time() - t0)

    def run_config(self, config: SweepConfig, det_gt: COCO, cls_gt: COCO) -> dict:
        """Evaluate a config using pre-computed embeddings. Only re-does matching."""
        key = (self._scales_key(config.scales), config.query_tta)
        per_image_data = self._cache[key]

        predictions = []

        for data in per_image_data:
            if data["embeddings"] is None:
                continue

            boxes = data["boxes"]
            det_scores = data["det_scores"]
            valid_indices = data["valid_indices"]
            embs = data["embeddings"]
            image_id = data["image_id"]

            # Filter by det_conf (higher than what we pre-computed at)
            # valid_indices maps embedding index → box index
            keep_mask = []
            for j, vi in enumerate(valid_indices):
                keep_mask.append(det_scores[vi] >= config.det_conf)

            if not any(keep_mask):
                continue

            # Filter embeddings and boxes
            kept_embs = embs[torch.tensor(keep_mask, dtype=torch.bool)]
            kept_vi = [vi for vi, keep in zip(valid_indices, keep_mask) if keep]

            # CAQE: expand embeddings with spatial neighbors
            if config.caqe_k > 0 and kept_embs.shape[0] > 1:
                kept_boxes = torch.from_numpy(boxes[np.array(kept_vi)]).to(kept_embs.device)
                kept_embs = apply_caqe(kept_embs, kept_boxes, k=config.caqe_k, alpha=config.caqe_alpha)

            # Classify (just matching — no DINOv2 forward pass)
            cat_ids, cos_scores = classify_knn(
                kept_embs, self._ref_embs, self._ref_ids, config.knn_k, config.unknown_threshold
            )

            # Build predictions
            for j, vi in enumerate(kept_vi):
                x1, y1, x2, y2 = boxes[vi]
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": cat_ids[j],
                        "bbox": [
                            round(float(x1), 2),
                            round(float(y1), 2),
                            round(float(x2 - x1), 2),
                            round(float(y2 - y1), 2),
                        ],
                        "score": round(float(det_scores[vi]) * cos_scores[j], 4),
                    }
                )

        return evaluate_coco_map(predictions, det_gt, cls_gt)


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------


def generate_configs(mode: str, axes: list = None) -> list:
    """Generate SweepConfig list based on mode."""
    if mode == "single_axis":
        configs = [SweepConfig()]  # baseline
        for axis_name, values in SWEEP_GRID.items():
            for val in values:
                kwargs = dict(BASELINE)
                kwargs[axis_name] = val
                cfg = SweepConfig(**kwargs)
                # Skip if identical to baseline
                if asdict(cfg) != asdict(configs[0]):
                    configs.append(cfg)
        return configs

    elif mode == "combinatorial":
        if not axes:
            axes = list(SWEEP_GRID.keys())
        # Build grid for selected axes only
        axis_values = {}
        for ax in axes:
            axis_values[ax] = SWEEP_GRID[ax]
        # Fill non-selected axes with baseline
        for ax in SWEEP_GRID:
            if ax not in axis_values:
                axis_values[ax] = [BASELINE[ax]]

        configs = []
        keys = list(axis_values.keys())
        for combo in itertools.product(*(axis_values[k] for k in keys)):
            kwargs = dict(zip(keys, combo))
            configs.append(SweepConfig(**kwargs))
        return configs

    else:
        raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_results_table(results: list):
    """Print formatted ASCII table sorted by combined score."""
    results = sorted(results, key=lambda r: r["metrics"]["combined"], reverse=True)

    header = (
        f"{'#':>3} | {'det_conf':>8} | {'scales':>17} | {'knn':>3} | {'q_tta':>5} | "
        f"{'unk_thr':>7} | {'caqe_k':>6} | {'caqe_a':>6} | "
        f"{'det_mAP50':>9} | {'cls_mAP50':>9} | {'combined':>8} | {'n_preds':>7}"
    )
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("Ablation Sweep Results")
    print("=" * len(header))
    print(header)
    print(sep)

    for i, r in enumerate(results):
        cfg = r["config"]
        m = r["metrics"]
        scales_str = str(cfg["scales"]).replace(" ", "")
        print(
            f"{i+1:>3} | {cfg['det_conf']:>8.2f} | {scales_str:>17} | {cfg['knn_k']:>3} | "
            f"{str(cfg['query_tta']):>5} | {cfg['unknown_threshold']:>7.2f} | "
            f"{cfg.get('caqe_k', 0):>6} | {cfg.get('caqe_alpha', 0.5):>6.1f} | "
            f"{m['det_mAP50']:>9.4f} | {m['cls_mAP50']:>9.4f} | {m['combined']:>8.4f} | "
            f"{m['n_preds']:>7}"
        )

    print(sep)
    best = results[0]
    print(f"\nBest combined: {best['metrics']['combined']:.4f}")
    print(f"  Config: {best['config']}")


def save_results_json(results: list, output_path: Path):
    """Save all results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved results to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Inference-time ablation sweep")
    p.add_argument("--mode", choices=["single_axis", "combinatorial"], default="single_axis")
    p.add_argument(
        "--axes", type=str, default=None, help="Comma-separated axes for combinatorial mode (e.g. det_conf,knn_k)"
    )
    p.add_argument("--yolo_weights", type=str, default="models/yolo_best.pt")
    p.add_argument("--classifier_weights", type=str, default="models/classifier_best.pt")
    p.add_argument("--ref_embeddings", type=str, default="models/ref_embeddings.pt")
    p.add_argument("--annotations", type=str, default="data/coco/train/annotations.json")
    p.add_argument("--val_dir", type=str, default="data/shelf_yolo/images/val")
    p.add_argument("--image_dir", type=str, default="data/coco/train/images")
    p.add_argument("--output", type=str, default="experiments/ablation_results.json")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- Load models ---
    from ultralytics import YOLO

    logger.info("Loading YOLO from %s", args.yolo_weights)
    yolo = YOLO(args.yolo_weights)

    logger.info("Loading classifier from %s", args.classifier_weights)
    cls_model = load_classifier(args.classifier_weights, device)

    logger.info("Loading reference embeddings from %s", args.ref_embeddings)
    ref_data = torch.load(args.ref_embeddings, map_location=device, weights_only=True)
    ref_embs = F.normalize(ref_data["embeddings"].to(device).float(), dim=1)
    ref_ids = ref_data["category_ids"].to(device)
    logger.info("Reference embeddings: %s, categories: %d", ref_embs.shape, len(ref_ids.unique()))

    # --- Build val set ---
    annotations_path = Path(args.annotations)
    val_dir = Path(args.val_dir)
    image_dir = Path(args.image_dir)

    val_image_ids = get_val_image_ids(annotations_path, val_dir)
    logger.info("Val images: %d", len(val_image_ids))

    det_gt, cls_gt = build_val_coco_gt(annotations_path, set(val_image_ids))

    # Map val image_ids back to actual image paths (in data/coco/train/images/)
    with open(annotations_path, "r") as f:
        coco_data = json.load(f)
    id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}
    val_image_paths = []
    for vid in val_image_ids:
        fname = id_to_filename[vid]
        img_path = image_dir / fname
        if img_path.exists():
            val_image_paths.append(img_path)
    logger.info("Val image paths resolved: %d", len(val_image_paths))

    # --- Detection cache ---
    det_cache = DetectionCache(yolo, device)

    # Pre-warm cache for all scales we might use
    all_scales = set()
    for vals in SWEEP_GRID["scales"]:
        if isinstance(vals, list):
            all_scales.update(vals)
        else:
            all_scales.add(vals)
    logger.info("Pre-caching detections for scales: %s", sorted(all_scales))
    t0 = time.time()
    for scale in sorted(all_scales):
        for img_path in val_image_paths:
            det_cache.get(img_path, scale)
    logger.info("Detection cache warmed in %.1fs (%d entries)", time.time() - t0, len(det_cache._cache))

    # --- Generate configs ---
    axes = args.axes.split(",") if args.axes else None
    configs = generate_configs(args.mode, axes)
    logger.info("Sweep mode: %s, configs: %d", args.mode, len(configs))

    # --- Pre-compute embeddings for unique (scales, query_tta) combos ---
    emb_store = EmbeddingStore(det_cache, cls_model, ref_embs, ref_ids, device, val_image_paths, args.batch_size)

    # Find minimum det_conf per (scales, tta) combo across all configs
    combo_min_conf = {}
    for cfg in configs:
        key = (tuple(cfg.scales), cfg.query_tta)
        if key not in combo_min_conf or cfg.det_conf < combo_min_conf[key]:
            combo_min_conf[key] = cfg.det_conf

    t_embed = time.time()
    for (scales_tuple, tta), min_conf in combo_min_conf.items():
        emb_store.precompute(list(scales_tuple), tta, min_conf)
    logger.info("All embeddings pre-computed in %.1fs", time.time() - t_embed)

    # --- Run sweep (matching only — no DINOv2 forward passes) ---
    results = []
    for i, cfg in enumerate(configs):
        t1 = time.time()
        metrics = emb_store.run_config(cfg, det_gt, cls_gt)
        elapsed = time.time() - t1
        result = {
            "config": asdict(cfg),
            "metrics": metrics,
            "elapsed_s": round(elapsed, 1),
        }
        results.append(result)
        logger.info(
            "[%d/%d] %s → det=%.4f cls=%.4f comb=%.4f (%.1fs)",
            i + 1,
            len(configs),
            cfg.label(),
            metrics["det_mAP50"],
            metrics["cls_mAP50"],
            metrics["combined"],
            elapsed,
        )

    # --- Output ---
    print_results_table(results)
    save_results_json(results, Path(args.output))

    total_time = sum(r["elapsed_s"] for r in results)
    logger.info("Total sweep time: %.1fs (+ pre-compute)", total_time)


if __name__ == "__main__":
    main()
