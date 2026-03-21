"""Mine shelf crops as supplementary reference embeddings.

For each category, selects the best shelf crops from the TRAIN split
and adds their embeddings to the reference gallery. This helps:

1. Categories with NO product reference images → now have refs (was 0% accuracy)
2. Categories with ambiguous product refs → domain-matched alternatives
3. All categories → shelf crops match inference conditions better than studio photos

Selection strategies:
  - centroid: top K closest to class centroid (most "typical" looking)
  - diverse: farthest-first traversal in embedding space (max coverage)

Train-only: uses image-level split (seed=42, 15% val) to prevent data leakage.

Usage:
    # Mine shelf refs and merge with product refs
    python -m vision_task.mine_shelf_refs \
        --checkpoint models/classifier_best.pt \
        --merge models/ref_embeddings.pt

    # Mine shelf refs only (no product refs)
    python -m vision_task.mine_shelf_refs \
        --checkpoint models/classifier_best.pt \
        --k_per_cat 10

    # More refs for no-ref categories, fewer for others
    python -m vision_task.mine_shelf_refs \
        --checkpoint models/classifier_best.pt \
        --k_per_cat 5 --k_no_ref 15 \
        --merge models/ref_embeddings.pt
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vision_task.config import CLASSIFIER_SIZE, CLASSIFIER_RESIZE
from vision_task.data.datasets import (
    PreExtractedCropDataset,
    TransformWrapper,
    get_train_val_indices,
)
from vision_task.data.transforms import get_eval_transform
from vision_task.embedder import GroceryEmbedder
from vision_task.evaluate import embed_dataset


logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description="Mine shelf crops as reference embeddings")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="models/classifier_best.pt",
        help="Path to trained classifier weights",
    )
    p.add_argument(
        "--output",
        type=str,
        default="models/ref_embeddings_with_shelf.pt",
        help="Output path for combined reference embeddings",
    )
    p.add_argument(
        "--merge",
        type=str,
        default=None,
        help="Merge with existing product ref embeddings (e.g. models/ref_embeddings.pt)",
    )
    p.add_argument(
        "--k_per_cat",
        type=int,
        default=5,
        help="Number of shelf crop refs to keep per category (for categories WITH product refs)",
    )
    p.add_argument(
        "--k_no_ref",
        type=int,
        default=15,
        help="Number of shelf crop refs for categories WITHOUT product refs (more needed since "
        "these are the ONLY refs)",
    )
    p.add_argument(
        "--selection",
        choices=["centroid", "diverse", "confirmed"],
        default="confirmed",
        help="Selection strategy: centroid (closest to shelf centroid), "
        "diverse (max coverage), confirmed (closest to product refs — best for "
        "categories WITH existing refs)",
    )
    p.add_argument(
        "--min_confirm_sim",
        type=float,
        default=0.5,
        help="Minimum cosine similarity to product ref for a shelf crop to be 'confirmed' "
        "(only used with --selection confirmed)",
    )
    p.add_argument(
        "--crops_dir",
        type=str,
        default="data/crops",
    )
    p.add_argument(
        "--manifest_path",
        type=str,
        default="data/crops/crops_manifest.csv",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--exclude_unknown",
        action="store_true",
        default=True,
        help="Exclude category 355 (unknown_product) from mined refs",
    )
    return p.parse_args()


def select_centroid(embeddings, k):
    """Select top K embeddings closest to class centroid."""
    centroid = F.normalize(embeddings.mean(dim=0, keepdim=True), dim=1)
    sims = (embeddings @ centroid.T).squeeze(1)
    k = min(k, embeddings.shape[0])
    _, top_indices = sims.topk(k)
    return embeddings[top_indices]


def select_diverse(embeddings, k):
    """Select K embeddings via farthest-first traversal (maximizes diversity)."""
    k = min(k, embeddings.shape[0])
    if k <= 1:
        return embeddings[:k]

    # Start with the one closest to centroid (most representative)
    centroid = F.normalize(embeddings.mean(dim=0, keepdim=True), dim=1)
    sims_to_centroid = (embeddings @ centroid.T).squeeze(1)
    first = sims_to_centroid.argmax().item()

    selected_indices = [first]
    for _ in range(k - 1):
        sel_embs = embeddings[selected_indices]
        # For each candidate, compute min similarity to any already-selected
        sims_to_selected = embeddings @ sel_embs.T  # [N, len(selected)]
        min_sims, _ = sims_to_selected.max(dim=1)  # Most similar selected neighbor
        # We want to pick the one LEAST similar to any selected (farthest)
        min_sims[selected_indices] = float("inf")  # Exclude already selected
        next_idx = min_sims.argmin().item()
        selected_indices.append(next_idx)

    return embeddings[selected_indices]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Load model ────────────────────────────────────────────────────────────
    checkpoint_path = Path(args.checkpoint)
    logger.info("Loading model from: %s", checkpoint_path)

    model = GroceryEmbedder(weights_path=None, freeze_backbone=True)
    sd = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    model = model.to(device).eval()

    # ── Load train shelf crops ────────────────────────────────────────────────
    full_ds = PreExtractedCropDataset(
        crops_dir=args.crops_dir,
        manifest_path=args.manifest_path,
        transform=None,  # We'll apply transform in the wrapper
        exclude_unknown=args.exclude_unknown,
    )
    train_indices, val_indices = get_train_val_indices(full_ds)
    train_ds = TransformWrapper(full_ds, train_indices, get_eval_transform())

    logger.info(
        "Train shelf crops: %d (from %d total, %d val excluded)",
        len(train_ds),
        len(full_ds),
        len(val_indices),
    )

    # ── Embed all train crops ─────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logger.info("Embedding all train crops...")
    all_embs, all_labels = embed_dataset(model, train_loader, device)
    all_embs = F.normalize(all_embs.float(), dim=1)
    logger.info("Embedded %d crops → %s", all_embs.shape[0], all_embs.shape)

    # ── Determine which categories have product refs ──────────────────────────
    product_ref_cats = set()
    if args.merge and Path(args.merge).exists():
        existing = torch.load(args.merge, map_location="cpu", weights_only=True)
        product_ref_cats = set(existing["category_ids"].unique().tolist())
        logger.info(
            "Product refs: %d embeddings across %d categories",
            existing["embeddings"].shape[0],
            len(product_ref_cats),
        )

    # ── Build per-category product ref centroids (for confirmed selection) ───
    ref_cat_embs = {}
    if args.merge and Path(args.merge).exists():
        for cat_id in product_ref_cats:
            mask = existing["category_ids"] == cat_id
            cat_refs = F.normalize(existing["embeddings"][mask].float(), dim=1)
            ref_cat_embs[cat_id] = cat_refs

    # ── Group by category and select ──────────────────────────────────────────
    cat_to_indices = defaultdict(list)
    for i, lbl in enumerate(all_labels.tolist()):
        cat_to_indices[lbl].append(i)

    fallback_selector = select_centroid if args.selection != "diverse" else select_diverse

    selected_embs = []
    selected_labels = []
    no_ref_cats = []
    with_ref_cats = []
    n_confirmed = 0
    n_fallback = 0

    for cat_id in sorted(cat_to_indices.keys()):
        indices = cat_to_indices[cat_id]
        cat_embs = all_embs[indices]

        has_product_refs = cat_id in product_ref_cats
        k = args.k_per_cat if has_product_refs else args.k_no_ref

        if args.selection == "confirmed" and has_product_refs and cat_id in ref_cat_embs:
            # CONFIRMED selection: score each shelf crop by max-sim to product refs
            # Only keep crops that the model confidently matches to this product
            cat_refs = ref_cat_embs[cat_id]  # [n_ref, 768]
            sims_to_refs = cat_embs @ cat_refs.T  # [n_shelf, n_ref]
            max_sims, _ = sims_to_refs.max(dim=1)  # best ref match per crop

            # Filter by confirmation threshold
            confirmed_mask = max_sims >= args.min_confirm_sim
            n_above = confirmed_mask.sum().item()

            if n_above >= max(1, k // 2):
                # Enough confirmed crops — use them, ranked by similarity
                confirmed_embs = cat_embs[confirmed_mask]
                confirmed_sims = max_sims[confirmed_mask]
                _, top_k = confirmed_sims.topk(min(k, n_above))
                chosen = confirmed_embs[top_k]
                n_confirmed += 1
            else:
                # Not enough confirmed — fall back to centroid selection
                chosen = fallback_selector(cat_embs, k)
                n_fallback += 1
        else:
            # No product refs or not using confirmed mode — use centroid/diverse
            chosen = fallback_selector(cat_embs, k)
            n_fallback += 1

        # Always add the shelf crop centroid as an extra ref
        # (robust to individual crop noise — represents "average shelf appearance")
        centroid = F.normalize(cat_embs.mean(dim=0, keepdim=True), dim=1)
        chosen = torch.cat([chosen, centroid], dim=0)

        selected_embs.append(chosen)
        selected_labels.extend([cat_id] * chosen.shape[0])

        if has_product_refs:
            with_ref_cats.append(cat_id)
        else:
            no_ref_cats.append(cat_id)

    shelf_embs = torch.cat(selected_embs, dim=0)
    shelf_labels = torch.tensor(selected_labels, dtype=torch.int64)

    logger.info(
        "Shelf-mined: %d embeddings from %d categories",
        shelf_embs.shape[0],
        len(cat_to_indices),
    )
    logger.info(
        "  Categories WITH product refs: %d (added %d shelf each)",
        len(with_ref_cats),
        args.k_per_cat,
    )
    logger.info(
        "  Categories WITHOUT product refs: %d (added %d shelf each): %s",
        len(no_ref_cats),
        args.k_no_ref,
        no_ref_cats,
    )
    if args.selection == "confirmed":
        logger.info(
            "  Confirmed selection: %d cats confirmed, %d cats fell back to centroid",
            n_confirmed,
            n_fallback,
        )

    # ── Merge with existing product refs ──────────────────────────────────────
    if args.merge and Path(args.merge).exists():
        combined_embs = torch.cat(
            [F.normalize(existing["embeddings"].float(), dim=1), shelf_embs], dim=0
        )
        combined_labels = torch.cat([existing["category_ids"], shelf_labels], dim=0)
        n_product = existing["embeddings"].shape[0]
    else:
        combined_embs = shelf_embs
        combined_labels = shelf_labels
        n_product = 0

    combined_embs = F.normalize(combined_embs, dim=1)

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "embeddings": combined_embs,
            "category_ids": combined_labels,
        },
        output_path,
    )

    n_total = combined_embs.shape[0]
    n_shelf = shelf_embs.shape[0]
    n_cats = len(combined_labels.unique())

    logger.info("")
    logger.info("=" * 60)
    logger.info("SAVED: %s", output_path)
    logger.info("  Total refs:   %d embeddings, %d categories", n_total, n_cats)
    logger.info("  Product refs: %d", n_product)
    logger.info("  Shelf-mined:  %d", n_shelf)
    logger.info(
        "  Size: %.2f MB (FP32)",
        n_total * 768 * 4 / 1024 / 1024,
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
