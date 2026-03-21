"""
Visualize predictions on shelf images with bounding boxes and category labels.

Usage:
    python -m vision_task.visualize \
        --predictions experiments/test_predictions.json \
        --image_dir data/coco/train/images \
        --output_dir experiments/visualizations \
        --max_images 10

    # Show only high-confidence predictions:
    python -m vision_task.visualize --predictions pred.json --score_thresh 0.3
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "data" / "coco" / "train" / "annotations.json"


def load_category_names(annotations_path):
    """Load category id -> name mapping from COCO annotations."""
    with open(annotations_path) as f:
        coco = json.load(f)
    return {c["id"]: c["name"] for c in coco["categories"]}


def load_image_map(annotations_path):
    """Load image id -> filename mapping from COCO annotations."""
    with open(annotations_path) as f:
        coco = json.load(f)
    return {img["id"]: img["file_name"] for img in coco["images"]}


def pick_color(cat_id):
    """Deterministic color per category."""
    r = (cat_id * 47 + 85) % 256
    g = (cat_id * 97 + 45) % 256
    b = (cat_id * 157 + 120) % 256
    # Ensure not too dark
    if r + g + b < 200:
        r = min(255, r + 80)
        g = min(255, g + 80)
    return (r, g, b)


def get_font(size=13):
    """Try to load a TrueType font, fall back to default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def draw_predictions(image, preds, category_names, score_thresh=0.0):
    """Draw bounding boxes and labels on a copy of the image."""
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = get_font(13)

    for pred in sorted(preds, key=lambda p: p["score"]):
        if pred["score"] < score_thresh:
            continue

        x, y, w, h = pred["bbox"]
        cat_id = pred["category_id"]
        score = pred["score"]
        color = pick_color(cat_id)

        # Box
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)

        # Label
        name = category_names.get(cat_id, f"cat_{cat_id}")
        short_name = name[:20] + ".." if len(name) > 20 else name
        label = f"{short_name} {score:.2f}"

        # Background rectangle for label
        lx, ly = x, max(0, y - 16)
        bbox = draw.textbbox((lx, ly), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((lx, ly), label, fill="white", font=font)

    return image


def main():
    parser = argparse.ArgumentParser(description="Visualize shelf predictions")
    parser.add_argument("--predictions", type=str, default="predictions.json", help="Path to predictions JSON")
    parser.add_argument("--image_dir", type=str, default="data/coco/train/images", help="Directory with shelf images")
    parser.add_argument(
        "--annotations", type=str, default=str(ANNOTATIONS), help="COCO annotations JSON for category names"
    )
    parser.add_argument(
        "--output_dir", type=str, default="experiments/visualizations", help="Output directory for annotated images"
    )
    parser.add_argument("--max_images", type=int, default=10, help="Max images to visualize (0 = all)")
    parser.add_argument("--score_thresh", type=float, default=0.1, help="Min score to draw a prediction")
    args = parser.parse_args()

    # Load predictions
    with open(args.predictions) as f:
        predictions = json.load(f)

    category_names = load_category_names(args.annotations)
    image_map = load_image_map(args.annotations)

    # Group predictions by image_id
    by_image = defaultdict(list)
    for pred in predictions:
        by_image[pred["image_id"]].append(pred)

    print(f"Loaded {len(predictions)} predictions across {len(by_image)} images")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir)

    count = 0
    for image_id in sorted(by_image.keys()):
        if args.max_images > 0 and count >= args.max_images:
            break

        # Find filename
        filename = image_map.get(image_id)
        if filename is None:
            # Try matching digits in filenames
            for p in image_dir.iterdir():
                digits = "".join(c for c in p.stem if c.isdigit())
                if digits and int(digits) == image_id:
                    filename = p.name
                    break

        if filename is None:
            print(f"  Warning: no image for id={image_id}, skipping")
            continue

        img_path = image_dir / filename
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        preds = by_image[image_id]
        annotated = draw_predictions(image, preds, category_names, args.score_thresh)

        out_path = output_dir / f"vis_{filename}"
        annotated.save(str(out_path), quality=90)

        n_drawn = len([p for p in preds if p["score"] >= args.score_thresh])
        print(f"  {out_path.name}: {n_drawn} predictions drawn")
        count += 1

    print(f"\nSaved {count} visualizations to {output_dir}")


if __name__ == "__main__":
    main()
