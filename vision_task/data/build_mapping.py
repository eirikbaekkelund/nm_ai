"""
Build category_id → product_code mapping via exact product name matching.

Reads COCO categories and metadata.json, matches by name, outputs
data/category_mapping.json for use by the classifier data pipeline.

Usage: python -m vision_task.data.build_mapping
"""

import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def build_mapping(coco_json, metadata_json, output_path):
    with open(coco_json, encoding="utf-8") as f:
        coco = json.load(f)

    with open(metadata_json, encoding="utf-8") as f:
        metadata = json.load(f)

    # Index metadata products by name for matching
    name_to_product = {}
    for prod in metadata["products"]:
        name_to_product[prod["product_name"]] = prod

    matched = []
    unmatched_categories = []
    unknown_category_id = None

    for cat in coco["categories"]:
        cat_id = cat["id"]
        cat_name = cat["name"]

        if cat_name == "unknown_product":
            unknown_category_id = cat_id
            continue

        product = name_to_product.get(cat_name)
        if product is not None:
            matched.append(
                {
                    "category_id": cat_id,
                    "product_code": product["product_code"],
                    "product_name": cat_name,
                    "has_images": product["has_images"],
                }
            )
        else:
            unmatched_categories.append(cat_id)

    # Detect CUSTOM_xxx dirs
    product_images_dir = Path(metadata_json).parent
    custom_dirs = sorted([d.name for d in product_images_dir.iterdir() if d.is_dir() and d.name.startswith("CUSTOM_")])

    mapping = {
        "matched": matched,
        "unmatched_categories": unmatched_categories,
        "unknown_category_id": unknown_category_id,
        "custom_dirs": custom_dirs,
        "stats": {
            "total_categories": len(coco["categories"]),
            "matched": len(matched),
            "unmatched": len(unmatched_categories),
            "unknown": 1 if unknown_category_id is not None else 0,
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    return mapping


def main():
    root = Path(__file__).resolve().parents[2]
    coco_json = root / "data" / "coco" / "train" / "annotations.json"
    metadata_json = root / "data" / "product_images" / "metadata.json"
    output_path = root / "data" / "category_mapping.json"

    print(f"COCO annotations: {coco_json}")
    print(f"Metadata: {metadata_json}")

    mapping = build_mapping(coco_json, metadata_json, output_path)
    stats = mapping["stats"]

    print(f"\n=== Mapping Results ===")
    print(f"Total COCO categories: {stats['total_categories']}")
    print(f"Matched to product_code: {stats['matched']}")
    print(f"Unmatched (no reference images): {stats['unmatched']}")
    print(f"Unknown product category: {mapping['unknown_category_id']}")
    print(f"CUSTOM dirs found: {len(mapping['custom_dirs'])}")

    # Print matched with/without images
    with_images = sum(1 for m in mapping["matched"] if m["has_images"])
    without_images = sum(1 for m in mapping["matched"] if not m["has_images"])
    print(f"\nOf {stats['matched']} matched:")
    print(f"  With reference images: {with_images}")
    print(f"  Without reference images: {without_images}")

    if mapping["unmatched_categories"]:
        print(f"\nUnmatched category IDs: {mapping['unmatched_categories']}")

    if mapping["custom_dirs"]:
        print(f"CUSTOM dirs (no mapping): {mapping['custom_dirs']}")

    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
