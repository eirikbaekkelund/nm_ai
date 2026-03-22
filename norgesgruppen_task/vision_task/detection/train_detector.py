"""
Two-stage YOLO detection training (supports YOLO11x, YOLO26x, YOLO26x-p2, RT-DETR):
  Stage 1: SKU-110K pretraining (optional, ~1.7M shelf annotations)
  Stage 2: Shelf fine-tuning (248 images, 22K annotations)

Usage:
  python -m vision_task.detection.train_detector                    # Full chain (YOLO11x)
  python -m vision_task.detection.train_detector --skip_sku110k     # Direct fine-tune
  python -m vision_task.detection.train_detector --skip_sku110k --base_model yolo26x.pt   # YOLO26x
  python -m vision_task.detection.train_detector --skip_sku110k --base_model yolo26x-p2.pt  # YOLO26x with P2 head
  python -m vision_task.detection.train_detector --shelf_epochs 2 --batch_shelf 2  # Quick test
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO

from vision_task.config import DETECTOR_IMGSZ

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
EXPERIMENTS_DIR = ROOT / "experiments" / "detection"


def resolve_yaml(yaml_path):
    """Create a temp YAML with absolute `path` — Ultralytics resolves relative
    paths against its own datasets_dir, not the YAML location."""
    yaml_path = Path(yaml_path).resolve()
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if "path" in cfg and not Path(cfg["path"]).is_absolute():
        cfg["path"] = str((yaml_path.parent / cfg["path"]).resolve())
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, dir=str(EXPERIMENTS_DIR))
    yaml.dump(cfg, tmp)
    tmp.close()
    return tmp.name


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO detector (11x, 26x, 26x-p2, RT-DETR)")
    p.add_argument("--skip_sku110k", action="store_true", help="Skip SKU-110K pretraining, fine-tune from COCO weights")
    p.add_argument("--sku110k_epochs", type=int, default=50)
    p.add_argument("--shelf_epochs", type=int, default=30)
    p.add_argument("--batch_sku", type=int, default=16, help="Batch size for SKU-110K (H100: 16, L4: 4)")
    p.add_argument("--batch_shelf", type=int, default=8, help="Batch size for shelf fine-tune")
    p.add_argument("--imgsz", type=int, default=DETECTOR_IMGSZ)
    p.add_argument("--device", default="0")
    p.add_argument(
        "--base_model",
        default="yolo11x.pt",
        help="Starting weights (yolo11x.pt, yolo26x.pt, yolo26x-p2.pt, rtdetr-x.pt)",
    )
    p.add_argument(
        "--sku110k_subset",
        type=float,
        default=1.0,
        help="Fraction of SKU-110K images to use (0.5 = 50%%). Saves disk space.",
    )
    return p.parse_args()


def _apply_sku110k_subset(sku_yolo_dir, fraction):
    """Subsample SKU-110K training images by moving extras to a _held_out dir.

    Only affects the 'train' split. Val is kept intact for fair evaluation.
    """
    import os
    import random

    train_img_dir = sku_yolo_dir / "images" / "train"
    train_lbl_dir = sku_yolo_dir / "labels" / "train"
    held_img_dir = sku_yolo_dir / "images" / "_train_held_out"
    held_lbl_dir = sku_yolo_dir / "labels" / "_train_held_out"

    # Restore any previously held-out images first
    for held, orig in [(held_img_dir, train_img_dir), (held_lbl_dir, train_lbl_dir)]:
        if held.exists():
            for f in held.iterdir():
                dest = orig / f.name
                if not dest.exists():
                    os.rename(str(f), str(dest))
            # Clean up empty dirs
            try:
                held.rmdir()
            except OSError:
                pass

    # Now subsample
    images = sorted(f for f in train_img_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    n_keep = max(1, int(len(images) * fraction))
    n_remove = len(images) - n_keep

    if n_remove <= 0:
        return

    rng = random.Random(42)
    to_remove = set(rng.sample(range(len(images)), n_remove))

    held_img_dir.mkdir(parents=True, exist_ok=True)
    held_lbl_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for i in to_remove:
        img_path = images[i]
        lbl_path = train_lbl_dir / (img_path.stem + ".txt")
        os.rename(str(img_path), str(held_img_dir / img_path.name))
        if lbl_path.exists():
            os.rename(str(lbl_path), str(held_lbl_dir / lbl_path.name))
        moved += 1

    print(f"SKU-110K subset: kept {n_keep}/{len(images)} images, moved {moved} to _held_out")


def train_sku110k(args):
    """Stage 1: Pretrain on SKU-110K shelf images."""
    sku_yolo_dir = ROOT / "data" / "sku110k_yolo"
    if not sku_yolo_dir.exists():
        print(f"ERROR: SKU-110K YOLO data not found at {sku_yolo_dir}")
        print("Run: python -m vision_task.detection.convert_sku110k")
        sys.exit(1)

    # Subset: remove random images from training split if requested
    if args.sku110k_subset < 1.0:
        _apply_sku110k_subset(sku_yolo_dir, args.sku110k_subset)

    yaml_path = resolve_yaml(Path(__file__).parent / "sku110k.yaml")
    print(f"\n{'='*60}")
    print(f"Stage 1: SKU-110K Pretraining")
    print(f"  Model:  {args.base_model}")
    print(f"  Data:   {yaml_path}")
    print(f"  Epochs: {args.sku110k_epochs}")
    print(f"  Batch:  {args.batch_sku}")
    print(f"  Imgsz:  {args.imgsz}")
    if args.sku110k_subset < 1.0:
        print(f"  Subset: {args.sku110k_subset*100:.0f}% of images")
    print(f"{'='*60}\n")

    model = YOLO(args.base_model)
    results = model.train(
        data=yaml_path,
        epochs=args.sku110k_epochs,
        imgsz=args.imgsz,
        batch=args.batch_sku,
        device=args.device,
        project=str(EXPERIMENTS_DIR),
        name="sku110k_pretrain",
        exist_ok=True,
        optimizer="SGD",
        # Keep default lr0=0.01 for pretraining
        mosaic=1.0,
        mixup=0.3,
        plots=False,
        verbose=True,
    )

    best_pt = EXPERIMENTS_DIR / "sku110k_pretrain" / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"ERROR: Expected weights not found at {best_pt}")
        sys.exit(1)

    print_results("SKU-110K Pretrain", results)
    return str(best_pt)


def train_shelf(args, base_weights):
    """Stage 2: Fine-tune on shelf data."""
    yaml_path = resolve_yaml(Path(__file__).parent / "shelf.yaml")
    print(f"\n{'='*60}")
    print(f"Stage 2: Shelf Fine-Tuning")
    print(f"  Model:  {base_weights}")
    print(f"  Data:   {yaml_path}")
    print(f"  Epochs: {args.shelf_epochs}")
    print(f"  Batch:  {args.batch_shelf}")
    print(f"  Imgsz:  {args.imgsz}")
    print(f"  LR:     0.001 (reduced for fine-tuning)")
    print(f"{'='*60}\n")

    model = YOLO(base_weights)
    results = model.train(
        data=yaml_path,
        epochs=args.shelf_epochs,
        imgsz=args.imgsz,
        batch=args.batch_shelf,
        device=args.device,
        project=str(EXPERIMENTS_DIR),
        name="shelf_finetune",
        exist_ok=True,
        optimizer="SGD",
        lr0=0.001,
        mosaic=1.0,
        mixup=0.3,
        plots=False,
        verbose=True,
    )

    best_pt = EXPERIMENTS_DIR / "shelf_finetune" / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"ERROR: Expected weights not found at {best_pt}")
        sys.exit(1)

    print_results("Shelf Fine-Tune", results)
    return str(best_pt)


def print_results(stage_name, results):
    """Print mAP and recall summary from Ultralytics results."""
    print(f"\n--- {stage_name} Results ---")
    if results and hasattr(results, "results_dict"):
        rd = results.results_dict
        for key in ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/recall(B)", "metrics/precision(B)"]:
            if key in rd:
                print(f"  {key}: {rd[key]:.4f}")
    else:
        print("  (results not available — check train logs)")


def main():
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Stage 1: SKU-110K pretrain (optional)
    if args.skip_sku110k:
        print("Skipping SKU-110K pretraining (--skip_sku110k)")
        base_weights = args.base_model
    else:
        base_weights = train_sku110k(args)

    # Stage 2: Shelf fine-tune
    best_pt = train_shelf(args, base_weights)

    # Copy best weights to models/
    dst = MODELS_DIR / "yolo_best.pt"
    shutil.copy2(best_pt, dst)
    print(f"\nBest weights copied to: {dst}")

    print(f"\n{'='*60}")
    print("Detection training complete!")
    print(f"  Weights: {dst}")
    print(f"  Logs:    {EXPERIMENTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
