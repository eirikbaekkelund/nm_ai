"""
Build competition submission .zip.

Usage:
    python build_submission.py                          # Default: submission.zip
    python build_submission.py --output my_submit.zip   # Custom name
    python build_submission.py --fp16                    # Convert classifier to FP16 (smaller)

Creates a minimal .zip with only files needed for L4 sandbox inference:
  - run.py              (entry point)
  - models/yolo_best.pt
  - models/classifier_best.pt  (stripped to model weights only)
  - models/ref_embeddings.pt
  - vision_task/__init__.py
  - vision_task/config.py
  - vision_task/embedder.py

Note: dinov2_vitb14.pth is NOT included — classifier_best.pt already
contains the full fine-tuned backbone weights.
"""

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent

# Files to bundle (relative to ROOT)
BUNDLE_FILES = [
    "run.py",
    "vision_task/__init__.py",
    "vision_task/config.py",
    "vision_task/embedder.py",
]

MODEL_FILES = [
    "models/yolo_best.pt",
    "models/ref_embeddings.pt",
]

# Classifier checkpoint gets special handling (strip + optional FP16)
CLASSIFIER_SRC = "models/classifier_best.pt"
CLASSIFIER_DST = "models/classifier_best.pt"  # name inside zip


def strip_checkpoint(src_path, fp16=False):
    """Strip checkpoint to model_state_dict only. Optionally convert to FP16."""
    ckpt = torch.load(src_path, map_location="cpu", weights_only=True)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        # Assume the whole file IS the state dict
        state_dict = ckpt

    if fp16:
        state_dict = {k: v.half() for k, v in state_dict.items()}

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.save({"model_state_dict": state_dict}, tmp.name)
    tmp.close()
    return Path(tmp.name)


def main():
    parser = argparse.ArgumentParser(description="Build competition submission .zip")
    parser.add_argument("--output", type=str, default="submission.zip")
    parser.add_argument("--fp16", action="store_true",
                        help="Convert classifier weights to FP16 (smaller file)")
    args = parser.parse_args()

    # Verify all files exist
    missing = []
    for f in BUNDLE_FILES + MODEL_FILES + [CLASSIFIER_SRC]:
        if not (ROOT / f).exists():
            missing.append(f)

    if missing:
        print("ERROR: Missing files:")
        for f in missing:
            print(f"  {f}")
        sys.exit(1)

    # Strip classifier checkpoint
    src = ROOT / CLASSIFIER_SRC
    orig_size = src.stat().st_size
    print(f"Stripping classifier checkpoint{' (FP16)' if args.fp16 else ''}...")
    slim_path = strip_checkpoint(src, fp16=args.fp16)
    slim_size = slim_path.stat().st_size
    print(f"  {orig_size / 1e6:.1f} MB -> {slim_size / 1e6:.1f} MB")

    # Build zip
    output = Path(args.output)
    print(f"\nBuilding {output}...")

    with zipfile.ZipFile(output, "w", zipfile.ZIP_STORED) as zf:
        # Code files
        for f in BUNDLE_FILES:
            src_file = ROOT / f
            zf.write(src_file, f)
            print(f"  + {f} ({src_file.stat().st_size / 1e3:.1f} KB)")

        # Model files (as-is)
        for f in MODEL_FILES:
            src_file = ROOT / f
            zf.write(src_file, f)
            print(f"  + {f} ({src_file.stat().st_size / 1e6:.1f} MB)")

        # Stripped classifier
        zf.write(slim_path, CLASSIFIER_DST)
        print(f"  + {CLASSIFIER_DST} ({slim_size / 1e6:.1f} MB)")

    # Cleanup temp file
    slim_path.unlink()

    total = output.stat().st_size
    print(f"\nDone! {output} ({total / 1e6:.1f} MB)")
    print(f"\nTo test locally:")
    print(f"  python run.py --input_dir data/coco/train/images --output_file predictions.json")


if __name__ == "__main__":
    main()
