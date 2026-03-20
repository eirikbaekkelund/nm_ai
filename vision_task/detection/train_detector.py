"""
Two-stage YOLO11x detection training:
  Stage 1: SKU-110K pretraining (optional, ~1.7M shelf annotations)
  Stage 2: Shelf fine-tuning (248 images, 22K annotations)

Usage:
  python -m vision_task.detection.train_detector                    # Full chain
  python -m vision_task.detection.train_detector --skip_sku110k     # Direct fine-tune
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
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=str(EXPERIMENTS_DIR)
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return tmp.name


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO11x detector")
    p.add_argument("--skip_sku110k", action="store_true",
                    help="Skip SKU-110K pretraining, fine-tune from COCO weights")
    p.add_argument("--sku110k_epochs", type=int, default=50)
    p.add_argument("--shelf_epochs", type=int, default=30)
    p.add_argument("--batch_sku", type=int, default=16,
                    help="Batch size for SKU-110K (H100: 16, L4: 4)")
    p.add_argument("--batch_shelf", type=int, default=8,
                    help="Batch size for shelf fine-tune")
    p.add_argument("--imgsz", type=int, default=DETECTOR_IMGSZ)
    p.add_argument("--device", default="0")
    p.add_argument("--base_model", default="yolo11x.pt",
                    help="Starting COCO-pretrained weights")
    return p.parse_args()


def train_sku110k(args):
    """Stage 1: Pretrain on SKU-110K shelf images."""
    sku_yolo_dir = ROOT / "data" / "sku110k_yolo"
    if not sku_yolo_dir.exists():
        print(f"ERROR: SKU-110K YOLO data not found at {sku_yolo_dir}")
        print("Run: python -m vision_task.detection.convert_sku110k")
        sys.exit(1)

    yaml_path = resolve_yaml(Path(__file__).parent / "sku110k.yaml")
    print(f"\n{'='*60}")
    print(f"Stage 1: SKU-110K Pretraining")
    print(f"  Model:  {args.base_model}")
    print(f"  Data:   {yaml_path}")
    print(f"  Epochs: {args.sku110k_epochs}")
    print(f"  Batch:  {args.batch_sku}")
    print(f"  Imgsz:  {args.imgsz}")
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
        # Augmentation (detector.md)
        mosaic=1.0,
        mixup=0.3,
        # Keep default lr0=0.01 for pretraining
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
        # Lower LR for fine-tuning (detector.md)
        lr0=0.001,
        # Augmentation (detector.md)
        mosaic=1.0,
        mixup=0.3,
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
        for key in ["metrics/mAP50(B)", "metrics/mAP50-95(B)",
                     "metrics/recall(B)", "metrics/precision(B)"]:
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
