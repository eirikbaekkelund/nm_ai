"""Export YOLO to ONNX and strip classifier for sandbox submission.

YOLO: Exported to FP16 ONNX — faster inference + smaller zip (~115 MB vs ~228 MB).
Classifier: Stripped to FP16 state_dict .pt — upcast to FP32 at load time.
       FP16 on disk to fit the 420 MB zip limit (~172 MB vs ~344 MB).

Run on training machine (requires ultralytics + CUDA):
    python export_models.py
    python export_models.py --tag classifier  # -> classifier.pt

Creates:
    models/yolo.onnx          (~115 MB FP16)
    models/classifier.pt      (~173 MB FP16 state_dict)
"""

import argparse
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Export models for submission")
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Name tag for exported files (e.g. 'phase7' -> classifier_phase7.pt)",
    )
    parser.add_argument(
        "--src",
        type=str,
        default=None,
        help="Path to classifier checkpoint (default: models/classifier_best.pt)",
    )
    parser.add_argument("--skip_yolo", action="store_true", help="Skip YOLO export")
    return parser.parse_args()


def export_yolo():
    """Export YOLO to ONNX FP16 — faster inference + smaller zip."""
    from ultralytics import YOLO

    src = ROOT / "models" / "yolo_best.pt"
    print(f"Exporting YOLO from {src}...")
    model = YOLO(str(src))
    device = 0 if torch.cuda.is_available() else "cpu"
    model.export(format="onnx", imgsz=1280, half=True, opset=17, device=device)

    # Ultralytics saves as yolo_best.onnx next to the source
    exported = src.with_suffix(".onnx")
    dst = ROOT / "models" / "yolo.onnx"
    if exported != dst and exported.exists():
        if dst.exists():
            dst.unlink()
        exported.rename(dst)
    print(f"  -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def strip_classifier(src_path=None, tag=None):
    """Strip classifier checkpoint to FP16 backbone-only state_dict.

    Input: src_path or models/classifier_best.pt (full training checkpoint with optimizer etc.)
    Output: models/classifier{_tag}.pt (FP16 state_dict matching GroceryEmbedder)
    """
    src = Path(src_path) if src_path else ROOT / "models" / "classifier_best.pt"
    suffix = f"_{tag}" if tag else ""
    dst = ROOT / "models" / f"classifier{suffix}.pt"
    print(f"Stripping classifier from {src}...")

    ckpt = torch.load(src, map_location="cpu", weights_only=True)

    # Auto-detect format: training checkpoint vs raw state_dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model_sd = ckpt["model_state_dict"]
        if "metrics" in ckpt:
            print(f"  Checkpoint metrics: {ckpt['metrics']}")
    else:
        model_sd = ckpt

    # Convert to FP16
    fp16_sd = {k: v.half() if v.is_floating_point() else v for k, v in model_sd.items()}

    torch.save(fp16_sd, dst)
    print(f"  -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    args = parse_args()
    if not args.skip_yolo:
        export_yolo()
    strip_classifier(src_path=args.src, tag=args.tag)
    print("\nDone! Files ready for build_submission.py")
