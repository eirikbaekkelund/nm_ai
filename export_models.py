"""Export YOLO and DINOv2 classifier to ONNX for sandbox submission.

The sandbox has ultralytics 8.1.0 (not 11.x) and timm 0.9.12 (not 1.x),
so .pt weights won't load. ONNX is the universal solution — sandbox has
onnxruntime-gpu 1.20.0 pre-installed.

Run on training machine (requires ultralytics + timm + CUDA):
    python export_models.py

Creates:
    models/yolo.onnx          (~109 MB FP16)
    models/classifier.onnx    (~174 MB FP16)
"""

import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def export_yolo():
    """Export YOLO11x to ONNX FP16."""
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


def export_classifier():
    """Export DINOv2+ArcFace classifier to ONNX FP16 with dynamic batch."""
    from vision_task.embedder import GroceryEmbedder

    src = ROOT / "models" / "classifier_best.pt"
    print(f"Exporting classifier from {src}...")

    model = GroceryEmbedder(weights_path=None, freeze_backbone=True)
    ckpt = torch.load(src, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.half().eval().cuda()

    dummy = torch.randn(1, 3, 518, 518, dtype=torch.float16, device="cuda")
    dst = ROOT / "models" / "classifier.onnx"

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(dst),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )

    print(f"  -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    export_yolo()
    export_classifier()
    print("\nDone! Files ready for build_submission.py")
