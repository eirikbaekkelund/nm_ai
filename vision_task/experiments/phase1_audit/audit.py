"""Phase 1 - Data Audit of COCO annotations and product reference images."""

import json
import sys
from collections import Counter
from pathlib import Path

# Force UTF-8 stdout on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COCO_JSON = ROOT / "data" / "coco" / "train" / "annotations.json"
IMAGE_DIR = ROOT / "data" / "coco" / "train" / "images"
PRODUCT_META = ROOT / "data" / "product_images" / "metadata.json"
OUT_DIR = Path(__file__).resolve().parent

# -- Load data ----------------------------------------------------------------
with open(COCO_JSON) as f:
    coco = json.load(f)

with open(PRODUCT_META) as f:
    product_meta = json.load(f)

images = {img["id"]: img for img in coco["images"]}
categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
annotations = coco["annotations"]

num_images = len(images)
num_annotations = len(annotations)
num_categories = len(categories)

# -- 1. Instances per class ----------------------------------------------------
class_counts = Counter(ann["category_id"] for ann in annotations)

# All 356 category IDs (0-355)
all_cat_ids = set(categories.keys())
annotated_cat_ids = set(class_counts.keys())
zero_example_ids = sorted(all_cat_ids - annotated_cat_ids)

# Build full table: category_id, name, count
class_table = []
for cid in sorted(categories.keys()):
    class_table.append(
        {
            "category_id": cid,
            "name": categories[cid],
            "annotation_count": class_counts.get(cid, 0),
        }
    )

# -- 2. Image dimension statistics ---------------------------------------------
widths = [img["width"] for img in images.values()]
heights = [img["height"] for img in images.values()]
areas = [w * h for w, h in zip(widths, heights)]

dim_stats = {
    "num_images": num_images,
    "width": {
        "min": min(widths),
        "max": max(widths),
        "mean": round(np.mean(widths), 1),
        "std": round(np.std(widths), 1),
    },
    "height": {
        "min": min(heights),
        "max": max(heights),
        "mean": round(np.mean(heights), 1),
        "std": round(np.std(heights), 1),
    },
    "unique_resolutions": len(set(zip(widths, heights))),
    "resolution_counts": dict(Counter(f"{w}x{h}" for w, h in zip(widths, heights)).most_common()),
}

# -- 3. Bbox statistics --------------------------------------------------------
bbox_widths = [ann["bbox"][2] for ann in annotations]
bbox_heights = [ann["bbox"][3] for ann in annotations]
bbox_areas = [w * h for w, h in zip(bbox_widths, bbox_heights)]

bbox_stats = {
    "num_annotations": num_annotations,
    "bbox_width": {
        "min": round(min(bbox_widths), 1),
        "max": round(max(bbox_widths), 1),
        "mean": round(np.mean(bbox_widths), 1),
    },
    "bbox_height": {
        "min": round(min(bbox_heights), 1),
        "max": round(max(bbox_heights), 1),
        "mean": round(np.mean(bbox_heights), 1),
    },
    "bbox_area": {
        "min": round(min(bbox_areas), 1),
        "max": round(max(bbox_areas), 1),
        "mean": round(np.mean(bbox_areas), 1),
    },
    "annotations_per_image": {"min": 0, "max": 0, "mean": 0},
}

anns_per_image = Counter(ann["image_id"] for ann in annotations)
anns_counts = list(anns_per_image.values())
bbox_stats["annotations_per_image"] = {
    "min": min(anns_counts),
    "max": max(anns_counts),
    "mean": round(np.mean(anns_counts), 1),
    "std": round(np.std(anns_counts), 1),
}

# -- 4. Category-to-product mapping via metadata --------------------------------
# Build category_id -> product_code mapping from metadata
cat_name_to_id = {cat["name"]: cat["id"] for cat in coco["categories"]}
product_codes_in_refs = set(p["product_code"] for p in product_meta["products"])
products_with_images = [p for p in product_meta["products"] if p["has_images"]]
products_without_images = [p for p in product_meta["products"] if not p["has_images"]]

# -- 5. Compile summary report -------------------------------------------------
counts_array = [class_counts.get(cid, 0) for cid in sorted(categories.keys())]
nonzero_counts = [c for c in counts_array if c > 0]

summary = {
    "dataset_overview": {
        "num_images": num_images,
        "num_annotations": num_annotations,
        "num_categories": num_categories,
    },
    "image_dimensions": dim_stats,
    "bbox_statistics": bbox_stats,
    "class_distribution": {
        "total_categories": num_categories,
        "categories_with_examples": len(annotated_cat_ids),
        "categories_with_zero_examples": len(zero_example_ids),
        "min_instances": min(counts_array),
        "max_instances": max(counts_array),
        "mean_instances": round(np.mean(counts_array), 1),
        "median_instances": round(float(np.median(counts_array)), 1),
        "std_instances": round(np.std(counts_array), 1),
        "nonzero_min": min(nonzero_counts) if nonzero_counts else 0,
        "nonzero_mean": round(np.mean(nonzero_counts), 1) if nonzero_counts else 0,
        "nonzero_median": round(float(np.median(nonzero_counts)), 1) if nonzero_counts else 0,
    },
    "reference_images": {
        "total_products_in_metadata": product_meta["total_products"],
        "products_with_images": product_meta["products_with_images"],
        "products_without_images": product_meta["products_without_images"],
        "total_reference_images": product_meta["total_images"],
        "products_missing_images": [
            {"product_code": p["product_code"], "name": p["product_name"]} for p in products_without_images
        ],
    },
    "zero_example_classes": [{"category_id": cid, "name": categories[cid]} for cid in zero_example_ids],
}

# ── 7. Save outputs ─────────────────────────────────────────────────────────
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

with open(OUT_DIR / "class_counts.json", "w") as f:
    json.dump(class_table, f, indent=2, ensure_ascii=False)

# ── 8. Plots ─────────────────────────────────────────────────────────────────

# 8a. Class frequency histogram (log scale)
fig, ax = plt.subplots(figsize=(14, 5))
sorted_counts = sorted(counts_array, reverse=True)
ax.bar(range(len(sorted_counts)), sorted_counts, width=1.0, color="steelblue", edgecolor="none")
ax.set_xlabel("Category (sorted by frequency)")
ax.set_ylabel("Number of annotations")
ax.set_title(f"Class Frequency Distribution — {num_categories} categories, {num_annotations} annotations")
ax.set_yscale("log")
ax.axhline(y=np.mean(counts_array), color="red", linestyle="--", label=f"Mean={np.mean(counts_array):.1f}")
ax.axhline(y=np.median(counts_array), color="orange", linestyle="--", label=f"Median={np.median(counts_array):.1f}")
ax.legend()
plt.tight_layout()
fig.savefig(OUT_DIR / "class_frequency_histogram.png", dpi=150)
plt.close(fig)

# 8b. Class frequency histogram (linear, zoomed to show long tail)
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(range(len(sorted_counts)), sorted_counts, width=1.0, color="steelblue", edgecolor="none")
ax.set_xlabel("Category (sorted by frequency)")
ax.set_ylabel("Number of annotations")
ax.set_title(f"Class Frequency Distribution (linear scale)")
ax.axhline(y=np.mean(counts_array), color="red", linestyle="--", label=f"Mean={np.mean(counts_array):.1f}")
ax.legend()
plt.tight_layout()
fig.savefig(OUT_DIR / "class_frequency_linear.png", dpi=150)
plt.close(fig)

# 8c. Bbox size scatter
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(bbox_widths, bbox_heights, s=1, alpha=0.3, color="steelblue")
ax.set_xlabel("Bbox width (px)")
ax.set_ylabel("Bbox height (px)")
ax.set_title(f"Bounding Box Sizes — {num_annotations} annotations")
ax.set_aspect("equal")
plt.tight_layout()
fig.savefig(OUT_DIR / "bbox_sizes.png", dpi=150)
plt.close(fig)

# 8d. Annotations per image histogram
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(anns_counts, bins=50, color="steelblue", edgecolor="white")
ax.set_xlabel("Annotations per image")
ax.set_ylabel("Number of images")
ax.set_title(f"Annotations per Image — {num_images} images")
ax.axvline(x=np.mean(anns_counts), color="red", linestyle="--", label=f"Mean={np.mean(anns_counts):.1f}")
ax.legend()
plt.tight_layout()
fig.savefig(OUT_DIR / "annotations_per_image.png", dpi=150)
plt.close(fig)

# 8e. Image dimension scatter
fig, ax = plt.subplots(figsize=(8, 6))
res_counter = Counter(zip(widths, heights))
for (w, h), count in res_counter.items():
    ax.scatter(w, h, s=count * 20, color="steelblue", alpha=0.7)
    ax.annotate(f"{w}x{h}\n(n={count})", (w, h), textcoords="offset points", xytext=(10, 5), fontsize=8)
ax.set_xlabel("Width (px)")
ax.set_ylabel("Height (px)")
ax.set_title("Image Resolutions")
plt.tight_layout()
fig.savefig(OUT_DIR / "image_resolutions.png", dpi=150)
plt.close(fig)

# -- 9. Print summary to console -----------------------------------------------
print("=" * 70)
print("PHASE 1 DATA AUDIT - SUMMARY")
print("=" * 70)
print(f"\nDataset: {num_images} images, {num_annotations} annotations, {num_categories} categories")

print(f"\n-- Image Dimensions --")
for key in ["width", "height"]:
    s = dim_stats[key]
    print(f"  {key}: min={s['min']}, max={s['max']}, mean={s['mean']}, std={s['std']}")
print(f"  Unique resolutions: {dim_stats['unique_resolutions']}")
for res, count in dim_stats["resolution_counts"].items():
    print(f"    {res}: {count} images")

print(f"\n-- Bounding Box Statistics --")
for key in ["bbox_width", "bbox_height", "bbox_area"]:
    s = bbox_stats[key]
    print(f"  {key}: min={s['min']}, max={s['max']}, mean={s['mean']}")
s = bbox_stats["annotations_per_image"]
print(f"  annotations/image: min={s['min']}, max={s['max']}, mean={s['mean']}, std={s['std']}")

print(f"\n-- Class Distribution --")
cd = summary["class_distribution"]
print(f"  Categories with examples: {cd['categories_with_examples']}/{cd['total_categories']}")
print(f"  Categories with ZERO examples: {cd['categories_with_zero_examples']}")
print(
    f"  Instances per class: min={cd['min_instances']}, max={cd['max_instances']}, "
    f"mean={cd['mean_instances']}, median={cd['median_instances']}"
)
print(f"  (nonzero only): min={cd['nonzero_min']}, mean={cd['nonzero_mean']}, median={cd['nonzero_median']}")

print(f"\n-- Reference Images --")
ri = summary["reference_images"]
print(f"  Products in metadata: {ri['total_products_in_metadata']}")
print(f"  Products with images: {ri['products_with_images']}")
print(f"  Products without images: {ri['products_without_images']}")
print(f"  Total reference images: {ri['total_reference_images']}")
if ri["products_missing_images"]:
    for p in ri["products_missing_images"]:
        print(f"    MISSING: {p['product_code']} - {p['name']}")

if zero_example_ids:
    print(f"\n-- Zero-Example Classes ({len(zero_example_ids)}) --")
    for cid in zero_example_ids:
        print(f"  [{cid}] {categories[cid]}")

# Top-10 and bottom-10 classes
print(f"\n-- Top 10 Most Common Classes --")
for cid, count in class_counts.most_common(10):
    print(f"  [{cid}] {categories[cid]}: {count}")

bottom_10 = sorted(
    [(cid, class_counts.get(cid, 0)) for cid in categories if class_counts.get(cid, 0) > 0], key=lambda x: x[1]
)[:10]
print(f"\n-- Bottom 10 Rarest Classes (nonzero) --")
for cid, count in bottom_10:
    print(f"  [{cid}] {categories[cid]}: {count}")

print(f"\nOutputs saved to: {OUT_DIR}")
print("  summary.json, class_counts.json")
print("  class_frequency_histogram.png, class_frequency_linear.png")
print("  bbox_sizes.png, annotations_per_image.png, image_resolutions.png")
