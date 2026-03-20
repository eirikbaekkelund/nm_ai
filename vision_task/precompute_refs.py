"""
Pre-compute L2-normalized reference embeddings for sandbox inference.

Loads trained DINOv2 from checkpoint, embeds all reference product images,
averages per-category, and saves to models/ref_embeddings.pt.

Usage:
    python -m vision_task.precompute_refs --checkpoint experiments/phase4_linear_probe/best.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vision_task.data.reference_dataset import ProductReferenceDataset
from vision_task.data.transforms import get_eval_transform
from vision_task.embedder import GroceryEmbedder
from vision_task.evaluate import aggregate_ref_embeddings, embed_dataset


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-compute reference embeddings")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/phase4_linear_probe/best.pt",
        help="Path to trained checkpoint (.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/ref_embeddings.pt",
        help="Output path for reference embeddings",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--product_images_dir",
        type=str,
        default="data/product_images",
    )
    parser.add_argument(
        "--mapping_path",
        type=str,
        default="data/category_mapping.json",
    )
    return parser.parse_args()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- Load model from checkpoint ---
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    model = GroceryEmbedder(freeze_backbone=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    if "metrics" in checkpoint:
        logger.info("Checkpoint metrics: %s", checkpoint["metrics"])

    # --- Load reference dataset ---
    ref_ds = ProductReferenceDataset(
        product_images_dir=args.product_images_dir,
        mapping_path=args.mapping_path,
        transform=get_eval_transform(),
    )
    ref_loader = DataLoader(
        ref_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logger.info("Reference images: %d across %d products", len(ref_ds), len(set(ref_ds.labels)))

    # --- Embed and aggregate ---
    ref_embs_raw, ref_labels_raw = embed_dataset(model, ref_loader, device)
    logger.info("Raw reference embeddings: %s", ref_embs_raw.shape)

    ref_embs, ref_labels = aggregate_ref_embeddings(ref_embs_raw, ref_labels_raw)
    logger.info("Aggregated: %d categories, embedding shape %s", len(ref_labels), ref_embs.shape)

    # --- Save ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "embeddings": ref_embs,  # [C, 768] L2-normalized
            "category_ids": ref_labels,  # [C] int64
        },
        output_path,
    )

    logger.info("Saved reference embeddings to %s", output_path)
    logger.info("Shape: %s, dtype: %s", ref_embs.shape, ref_embs.dtype)

    # Verify normalization
    norms = ref_embs.norm(dim=1)
    logger.info("Norm range: [%.6f, %.6f] (should be ~1.0)", norms.min(), norms.max())


if __name__ == "__main__":
    main()
