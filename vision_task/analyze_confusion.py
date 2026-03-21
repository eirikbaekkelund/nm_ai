"""
Analyze reference embedding similarities for confused category pairs.

Answers: are the confused products inherently indistinguishable in embedding space,
or is there signal that training/post-processing could exploit?

Usage:
    python -m vision_task.analyze_confusion
    python -m vision_task.analyze_confusion --diagnosis_path experiments/diagnostics/diagnosis_report.json
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Analyze ref embedding confusion")
    p.add_argument("--ref_embeddings", type=str, default="models/ref_embeddings.pt")
    p.add_argument("--diagnosis_path", type=str, default=None)
    p.add_argument("--annotations", type=str, default="data/coco/train/annotations.json")
    return p.parse_args()


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    # Load ref embeddings
    ref_data = torch.load(args.ref_embeddings, map_location="cpu", weights_only=True)
    ref_embs = F.normalize(ref_data["embeddings"].float(), dim=1)
    ref_ids = ref_data["category_ids"]
    logger.info("Ref embeddings: %s, %d unique categories", ref_embs.shape, len(ref_ids.unique()))

    # Count refs per category
    ref_counts = Counter(ref_ids.tolist())

    # Load category names
    with open(args.annotations, "r") as f:
        coco = json.load(f)
    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

    # Load confusion pairs
    if args.diagnosis_path and Path(args.diagnosis_path).exists():
        with open(args.diagnosis_path) as f:
            report = json.load(f)
        pairs = [(p["gt_cat"], p["pred_cat"], p["count"]) for p in report["top_confusion_pairs"]]
    else:
        from vision_task.hard_negatives import FALLBACK_CONFUSION_PAIRS
        pairs = FALLBACK_CONFUSION_PAIRS

    # ── Per-category self-similarity (intra-class) ──
    logger.info("\n" + "=" * 80)
    logger.info("INTRA-CLASS SIMILARITY (should be high — refs for same product)")
    logger.info("=" * 80)
    cat_centroids = {}
    for cat_id in ref_ids.unique().tolist():
        mask = ref_ids == cat_id
        embs = ref_embs[mask]
        cat_centroids[cat_id] = F.normalize(embs.mean(dim=0, keepdim=True), dim=1)
        if embs.shape[0] > 1:
            intra_sim = (embs @ embs.T)
            # Get off-diagonal elements
            n = embs.shape[0]
            off_diag = intra_sim[~torch.eye(n, dtype=torch.bool)].tolist()
            mean_intra = sum(off_diag) / len(off_diag)
        else:
            mean_intra = 1.0
        # Only print for confused categories
        for a, b, count in pairs[:20]:
            if cat_id in (a, b):
                logger.info("  cat=%3d (%d refs) intra_sim=%.4f  %s",
                            cat_id, embs.shape[0], mean_intra,
                            id_to_name.get(cat_id, "?")[:50])
                break

    # ── Cross-category similarity for confused pairs ──
    logger.info("\n" + "=" * 80)
    logger.info("CROSS-CATEGORY SIMILARITY (confused pairs — should be LOW if separable)")
    logger.info("=" * 80)
    logger.info(f"{'gt':>5} {'pred':>5} {'cnt':>4} | {'max_cross':>9} {'mean_cross':>10} {'margin':>7} | gt_name -> pred_name")
    logger.info("-" * 100)

    for gt_cat, pred_cat, count in pairs[:25]:
        mask_a = ref_ids == gt_cat
        mask_b = ref_ids == pred_cat
        n_a = mask_a.sum().item()
        n_b = mask_b.sum().item()

        if n_a == 0 or n_b == 0:
            logger.info(f"{gt_cat:>5} {pred_cat:>5} {count:>4} | {'NO REFS':>9} {'':>10} {'':>7} | "
                        f"{id_to_name.get(gt_cat, '?')[:30]} -> {id_to_name.get(pred_cat, '?')[:30]}")
            continue

        embs_a = ref_embs[mask_a]
        embs_b = ref_embs[mask_b]
        cross_sim = embs_a @ embs_b.T  # [n_a, n_b]

        max_cross = cross_sim.max().item()
        mean_cross = cross_sim.mean().item()

        # Margin: how much closer is same-class centroid vs cross-class?
        # For gt_cat: sim to own centroid vs sim to pred_cat centroid
        if gt_cat in cat_centroids and pred_cat in cat_centroids:
            self_sim = (embs_a @ cat_centroids[gt_cat].T).mean().item()
            cross_centroid = (embs_a @ cat_centroids[pred_cat].T).mean().item()
            margin = self_sim - cross_centroid
        else:
            margin = float("nan")

        logger.info(f"{gt_cat:>5} {pred_cat:>5} {count:>4} | {max_cross:>9.4f} {mean_cross:>10.4f} {margin:>7.4f} | "
                    f"{id_to_name.get(gt_cat, '?')[:30]} -> {id_to_name.get(pred_cat, '?')[:30]}")

    # ── Global stats: what's a "normal" cross-category similarity? ──
    logger.info("\n" + "=" * 80)
    logger.info("GLOBAL CROSS-CATEGORY SIMILARITY DISTRIBUTION")
    logger.info("=" * 80)

    # Sample 200 random non-confused category pairs for baseline
    unique_cats = ref_ids.unique().tolist()
    import random
    rng = random.Random(42)
    random_sims = []
    for _ in range(200):
        a, b = rng.sample(unique_cats, 2)
        mask_a = ref_ids == a
        mask_b = ref_ids == b
        if mask_a.sum() == 0 or mask_b.sum() == 0:
            continue
        embs_a = ref_embs[mask_a]
        embs_b = ref_embs[mask_b]
        sim = (embs_a @ embs_b.T).mean().item()
        random_sims.append(sim)

    if random_sims:
        random_sims.sort()
        logger.info("Random pair similarity: mean=%.4f, std=%.4f, p50=%.4f, p95=%.4f, max=%.4f",
                    sum(random_sims)/len(random_sims),
                    (sum((s - sum(random_sims)/len(random_sims))**2 for s in random_sims)/len(random_sims))**0.5,
                    random_sims[len(random_sims)//2],
                    random_sims[int(len(random_sims)*0.95)],
                    random_sims[-1])

    # ── Verdict ──
    logger.info("\n" + "=" * 80)
    logger.info("VERDICT: What should we do?")
    logger.info("=" * 80)
    logger.info("If max_cross > 0.85 for a pair → refs are inherently ambiguous.")
    logger.info("  → Hard-negative training alone won't fix it.")
    logger.info("  → Need: more ref angles, ref TTA, or confusion-aware post-processing.")
    logger.info("If margin < 0.05 → centroid separation is too thin for reliable matching.")
    logger.info("  → Need: push centroids apart via hard-negative training.")
    logger.info("If a category has NO REFS → completely unsolvable via embedding matching.")


if __name__ == "__main__":
    main()
