"""Export YOLO to ONNX and strip classifier for sandbox submission.

YOLO: Exported to FP32 ONNX (sandbox has ultralytics 8.1.0, not 11.x/26.x).
Classifier: Stripped to FP16 state_dict .pt — upcast to FP32 at load time.
       FP16 on disk to fit the 420 MB zip limit (~172 MB vs ~344 MB).

Run on training machine (requires ultralytics + CUDA):
    python export_models.py

Creates:
    models/yolo.onnx          (~228 MB FP32)
    models/classifier.pt      (~173 MB FP16 state_dict)
"""

import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def export_yolo():
    """Export YOLO to ONNX FP32."""
    from ultralytics import YOLO

    src = ROOT / "models" / "yolo_best.pt"
    print(f"Exporting YOLO from {src}...")
    model = YOLO(str(src))
    device = 0 if torch.cuda.is_available() else "cpu"
    model.export(format="onnx", imgsz=1280, half=False, opset=17, device=device)

    # Ultralytics saves as yolo_best.onnx next to the source
    exported = src.with_suffix(".onnx")
    dst = ROOT / "models" / "yolo.onnx"
    if exported != dst and exported.exists():
        if dst.exists():
            dst.unlink()
        exported.rename(dst)
    print(f"  -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def strip_classifier():
    """Strip classifier checkpoint to FP16 backbone-only state_dict.

    Input: models/classifier_best.pt (full training checkpoint with optimizer etc.)
    Output: models/classifier.pt (FP16 state_dict matching GroceryEmbedder)
    """
    src = ROOT / "models" / "classifier_best.pt"
    dst = ROOT / "models" / "classifier.pt"
    print(f"Stripping classifier from {src}...")

    ckpt = torch.load(src, map_location="cpu", weights_only=True)
    model_sd = ckpt["model_state_dict"]

    # Convert to FP16
    fp16_sd = {k: v.half() if v.is_floating_point() else v for k, v in model_sd.items()}

    torch.save(fp16_sd, dst)
    print(f"  -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    export_yolo()
    strip_classifier()
    print("\nDone! Files ready for build_submission.py")
