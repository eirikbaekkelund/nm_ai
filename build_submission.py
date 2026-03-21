"""Build competition submission .zip — YOLO ONNX + native DINOv2 classifier.

Usage:
    python build_submission.py                     # Default: submission.zip
    python build_submission.py --output my.zip     # Custom name

Prerequisites:
    Run export_models.py first to create model files.

Bundle contents (all at zip root — no subdirectories):
  - run.py                       (~8 KB)
  - embedder.py                  (~0.5 KB)
  - yolo.onnx                    (~109-115 MB FP16)
  - classifier.pt                (~173 MB FP16 state_dict)
  - ref_embeddings.pt            (~1 MB)
"""

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (source_path_relative_to_ROOT, name_in_zip)
FILES = [
    ("run.py", "run.py"),
    ("embedder.py", "embedder.py"),
    ("models/yolo.onnx", "yolo.onnx"),
    ("models/classifier.pt", "classifier.pt"),
    ("models/ref_embeddings.pt", "ref_embeddings.pt"),
]


def main():
    parser = argparse.ArgumentParser(description="Build competition submission .zip")
    parser.add_argument("--output", type=str, default="submission.zip")
    args = parser.parse_args()

    missing = [src for src, _ in FILES if not (ROOT / src).exists()]
    if missing:
        print("ERROR: Missing files (run export_models.py first?):")
        for f in missing:
            print(f"  {f}")
        sys.exit(1)

    output = Path(args.output)
    print(f"Building {output}...")

    with zipfile.ZipFile(output, "w", zipfile.ZIP_STORED) as zf:
        for src, arcname in FILES:
            src_path = ROOT / src
            zf.write(src_path, arcname)
            size = src_path.stat().st_size
            if size < 1e6:
                print(f"  + {arcname} ({size / 1e3:.1f} KB)")
            else:
                print(f"  + {arcname} ({size / 1e6:.1f} MB)")

    total = output.stat().st_size
    print(f"\nDone! {output} ({total / 1e6:.1f} MB)")
    if total > 420e6:
        print("  WARNING: Exceeds 420 MB limit!")
    else:
        print(f"  Under 420 MB limit ({420 - total / 1e6:.0f} MB headroom)")


if __name__ == "__main__":
    main()
