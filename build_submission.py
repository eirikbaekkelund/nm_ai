"""Build competition submission .zip — ONNX models, no vision_task package.

Usage:
    python build_submission.py                     # Default: submission.zip
    python build_submission.py --output my.zip     # Custom name

Prerequisites:
    Run export_models.py first to create ONNX files.

Bundle contents:
  - run.py                       (~6 KB)
  - models/yolo.onnx             (~109 MB FP16)
  - models/classifier.onnx       (~174 MB FP16)
  - models/ref_embeddings.pt     (~1 MB)
"""

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

FILES = [
    "run.py",
    "models/yolo.onnx",
    "models/classifier.onnx",
    "models/ref_embeddings.pt",
]


def main():
    parser = argparse.ArgumentParser(description="Build competition submission .zip")
    parser.add_argument("--output", type=str, default="submission.zip")
    args = parser.parse_args()

    missing = [f for f in FILES if not (ROOT / f).exists()]
    if missing:
        print("ERROR: Missing files (run export_models.py first?):")
        for f in missing:
            print(f"  {f}")
        sys.exit(1)

    output = Path(args.output)
    print(f"Building {output}...")

    with zipfile.ZipFile(output, "w", zipfile.ZIP_STORED) as zf:
        for f in FILES:
            src = ROOT / f
            zf.write(src, f)
            size = src.stat().st_size
            if size < 1e6:
                print(f"  + {f} ({size / 1e3:.1f} KB)")
            else:
                print(f"  + {f} ({size / 1e6:.1f} MB)")

    total = output.stat().st_size
    print(f"\nDone! {output} ({total / 1e6:.1f} MB)")
    if total > 420e6:
        print("  WARNING: Exceeds 420 MB limit!")
    else:
        print(f"  Under 420 MB limit ({420 - total / 1e6:.0f} MB headroom)")


if __name__ == "__main__":
    main()
