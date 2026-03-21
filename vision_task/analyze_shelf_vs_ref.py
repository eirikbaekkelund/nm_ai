"""Compare shelf crop embeddings vs product ref embeddings for confused pairs.

Key hypothesis: ArcFace training pushes shelf crop embeddings apart,
so shelf crop centroids may be MORE separable than product ref centroids
for confused pairs (e.g., EVERGOOD CLASSIC FILTER vs KOKMALT).

Also reports per-category stats useful for deciding the reference strategy.

Usage:
    python -m vision_task.analyze_shelf_vs_ref --checkpoint models/classifier_best.pt
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vision_task.data.datasets import (
    PreExtractedCropDataset,
    TransformWrapper,
    get_train_val_indices,
)
from vision_task.data.transforms import get_eval_transform
from vision_task.embedder import GroceryEmbedder
from vision_task.evaluate import embed_dataset
from vision_task.hard_negatives import FALLBACK_CONFUSION_PAIRS

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/classifier_best.pt")
    p.add_argument("--ref_embeddings", default="models/ref_embeddings.pt")
    p.add_argument("--annotations", default="data/coco/train/annotations.json")
    p.add_argument("--crops_dir", default="data/crops")
    p.add_argument("--manifest_path", default="data/crops/crops_manifest.csv")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load category names
    with open(args.annotations, "r") as f:
        coco = json.load(f)
    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

    # Load product ref embeddings
    ref_data = torch.load(args.ref_embeddings, map_location="cpu", weights_only=True)
    ref_embs = F.normalize(ref_data["embeddings"].float(), dim=1)
    ref_ids = ref_data["category_ids"]
    ref_cats = set(ref_ids.unique().tolist())
    logger.info("Product refs: %s, %d categories", ref_embs.shape, len(ref_cats))

    # Load model
    model = GroceryEmbedder(weights_path=None, freeze_backbone=True)
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    model = model.to(device).eval()

    # Embed TRAIN shelf crops
    full_ds = PreExtractedCropDataset(
        crops_dir=args.crops_dir,
        manifest_path=args.manifest_path,
        transform=None,
        exclude_unknown=True,
    )
    train_indices, _ = get_train_val_indices(full_ds)
    train_ds = TransformWrapper(full_ds, train_indices, get_eval_transform())
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    logger.info("Embedding %d train shelf crops...", len(train_ds))
    shelf_embs, shelf_labels = embed_dataset(model, train_loader, device)
    shelf_embs = F.normalize(shelf_embs.float(), dim=1)

    # Group shelf embeddings by category
    shelf_by_cat = defaultdict(list)
    for i, lbl in enumerate(shelf_labels.tolist()):
        shelf_by_cat[lbl].append(i)

    # Compute centroids for both domains
    ref_centroids = {}
    for cat_id in ref_ids.unique().tolist():
        mask = ref_ids == cat_id
        ref_centroids[cat_id] = F.normalize(ref_embs[mask].mean(dim=0, keepdim=True), dim=1)

    shelf_centroids = {}
    for cat_id, indices in shelf_by_cat.items():
        embs = shelf_embs[indices]
        shelf_centroids[cat_id] = F.normalize(embs.mean(dim=0, keepdim=True), dim=1)

    # ── Compare confused pairs ────────────────────────────────────────────────
    logger.info("\n" + "=" * 100)
    logger.info("CONFUSED PAIRS: Product ref centroid vs Shelf crop centroid separation")
    logger.info("=" * 100)
    logger.info(
        f"{'gt':>5} {'pred':>5} {'cnt':>4} | "
        f"{'ref_cross':>10} {'ref_margin':>11} | "
        f"{'shelf_cross':>12} {'shelf_margin':>13} | "
        f"{'improvement':>12} | gt_name → pred_name"
    )
    logger.info("-" * 130)

    for gt_cat, pred_cat, count in FALLBACK_CONFUSION_PAIRS:
        gt_name = id_to_name.get(gt_cat, "?")[:25]
        pred_name = id_to_name.get(pred_cat, "?")[:25]

        # Product ref comparison
        if gt_cat in ref_centroids and pred_cat in ref_centroids:
            ref_cross = (ref_centroids[gt_cat] @ ref_centroids[pred_cat].T).item()
            ref_self_a = 1.0  # centroid-to-self is always 1
            ref_margin = 1.0 - ref_cross
        else:
            ref_cross = float("nan")
            ref_margin = float("nan")

        # Shelf crop comparison
        if gt_cat in shelf_centroids and pred_cat in shelf_centroids:
            shelf_cross = (shelf_centroids[gt_cat] @ shelf_centroids[pred_cat].T).item()
            shelf_margin = 1.0 - shelf_cross
        else:
            shelf_cross = float("nan")
            shelf_margin = float("nan")

        # Improvement
        if ref_margin == ref_margin and shelf_margin == shelf_margin:  # not NaN
            improvement = shelf_margin - ref_margin
        else:
            improvement = float("nan")

        logger.info(
            f"{gt_cat:>5} {pred_cat:>5} {count:>4} | "
            f"{ref_cross:>10.4f} {ref_margin:>11.4f} | "
            f"{shelf_cross:>12.4f} {shelf_margin:>13.4f} | "
            f"{improvement:>+12.4f} | "
            f"{gt_name} → {pred_name}"
        )

    # ── Domain gap analysis ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 100)
    logger.info("DOMAIN GAP: Product ref centroid vs Shelf crop centroid (SAME category)")
    logger.info("=" * 100)
    logger.info(
        f"{'cat':>5} {'n_ref':>5} {'n_shelf':>7} | "
        f"{'ref↔shelf':>10} {'ref_intra':>10} {'shelf_intra':>12} | name"
    )
    logger.info("-" * 100)

    domain_gaps = []
    for cat_id in sorted(ref_centroids.keys()):
        if cat_id not in shelf_centroids:
            continue

        # Similarity between product ref centroid and shelf crop centroid
        cross_domain = (ref_centroids[cat_id] @ shelf_centroids[cat_id].T).item()
        domain_gaps.append(cross_domain)

        # Only print for confused categories + a sample of others
        n_ref = (ref_ids == cat_id).sum().item()
        n_shelf = len(shelf_by_cat.get(cat_id, []))

        # Intra-class similarity within each domain
        ref_mask = ref_ids == cat_id
        if ref_mask.sum() > 1:
            ref_e = ref_embs[ref_mask]
            sim = ref_e @ ref_e.T
            n = sim.shape[0]
            ref_intra = sim[~torch.eye(n, dtype=torch.bool)].mean().item()
        else:
            ref_intra = 1.0

        if n_shelf > 1:
            shelf_e = shelf_embs[shelf_by_cat[cat_id]]
            sim = shelf_e @ shelf_e.T
            n = sim.shape[0]
            shelf_intra = sim[~torch.eye(n, dtype=torch.bool)].mean().item()
        else:
            shelf_intra = 1.0

        # Print all — useful for understanding the full picture
        is_confused = any(cat_id in (a, b) for a, b, _ in FALLBACK_CONFUSION_PAIRS)
        marker = " ***" if is_confused else ""
        logger.info(
            f"{cat_id:>5} {n_ref:>5} {n_shelf:>7} | "
            f"{cross_domain:>10.4f} {ref_intra:>10.4f} {shelf_intra:>12.4f} | "
            f"{id_to_name.get(cat_id, '?')[:40]}{marker}"
        )

    # Summary stats
    if domain_gaps:
        domain_gaps.sort()
        logger.info("\n" + "=" * 100)
        logger.info("DOMAIN GAP SUMMARY (ref centroid ↔ shelf centroid, same category)")
        logger.info("=" * 100)
        logger.info(
            "  mean=%.4f, std=%.4f, min=%.4f, p25=%.4f, p50=%.4f, p75=%.4f, max=%.4f",
            sum(domain_gaps) / len(domain_gaps),
            (sum((g - sum(domain_gaps) / len(domain_gaps)) ** 2 for g in domain_gaps) / len(domain_gaps)) ** 0.5,
            domain_gaps[0],
            domain_gaps[len(domain_gaps) // 4],
            domain_gaps[len(domain_gaps) // 2],
            domain_gaps[3 * len(domain_gaps) // 4],
            domain_gaps[-1],
        )

    # ── No-ref categories ─────────────────────────────────────────────────────
    no_ref_cats = [c for c in shelf_by_cat if c not in ref_cats]
    if no_ref_cats:
        logger.info("\n" + "=" * 100)
        logger.info("NO-REF CATEGORIES (shelf crops are the ONLY option)")
        logger.info("=" * 100)
        logger.info(f"{'cat':>5} {'n_shelf':>7} {'shelf_intra':>12} | name")
        logger.info("-" * 80)
        for cat_id in sorted(no_ref_cats):
            indices = shelf_by_cat[cat_id]
            n_shelf = len(indices)
            embs = shelf_embs[indices]
            if n_shelf > 1:
                sim = embs @ embs.T
                n = sim.shape[0]
                intra = sim[~torch.eye(n, dtype=torch.bool)].mean().item()
            else:
                intra = 1.0
            logger.info(
                f"{cat_id:>5} {n_shelf:>7} {intra:>12.4f} | "
                f"{id_to_name.get(cat_id, '?')[:50]}"
            )

    # ── Recommendation ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 100)
    logger.info("STRATEGY RECOMMENDATION")
    logger.info("=" * 100)
    logger.info("If shelf_margin > ref_margin for confused pairs:")
    logger.info("  → Shelf crop centroids are BETTER references than product refs")
    logger.info("  → Use shelf centroids as primary + product refs as supplementary")
    logger.info("If ref↔shelf gap is small (>0.8):")
    logger.info("  → Low domain gap, product refs are decent proxies")
    logger.info("If ref↔shelf gap is large (<0.6):")
    logger.info("  → High domain gap, shelf crop refs strongly preferred")


if __name__ == "__main__":
    main()
