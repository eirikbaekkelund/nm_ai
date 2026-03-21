"""
Hard-Negative Mining — confusion pair definitions, sampler boost, contrastive loss.

Targets the 30% of classification errors caused by same-brand confusion
(e.g. ALI filter vs kok, EVERGOOD classic vs dark roast).

Training-only module — never included in submission.

Usage:
    from vision_task.hard_negatives import (
        load_confusion_pairs,
        HardNegativeSampler,
        hard_negative_loss,
    )
"""

import json
import math
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler


# ---------------------------------------------------------------------------
# Confusion pair loading
# ---------------------------------------------------------------------------


def load_confusion_pairs(diagnosis_path: str, min_count: int = 3) -> list:
    """Load confusion pairs from a diagnosis_report.json.

    Args:
        diagnosis_path: path to diagnosis_report.json from vision_task.diagnose
        min_count: minimum confusion count to include a pair

    Returns:
        List of (cat_a, cat_b, count) tuples, deduplicated and symmetric.
        Each pair appears once (smaller cat_id first).
    """
    path = Path(diagnosis_path)
    if not path.exists():
        raise FileNotFoundError(f"Diagnosis report not found: {path}")

    with open(path, "r") as f:
        report = json.load(f)

    raw_pairs = report.get("top_confusion_pairs", [])

    # Merge symmetric pairs: (a→b) and (b→a) into one entry
    merged = Counter()
    for entry in raw_pairs:
        a, b = entry["gt_cat"], entry["pred_cat"]
        key = (min(a, b), max(a, b))
        merged[key] += entry["count"]

    pairs = [
        (a, b, count)
        for (a, b), count in merged.items()
        if count >= min_count
    ]
    pairs.sort(key=lambda x: -x[2])
    return pairs


# Fallback: hardcoded top confusion pairs from typical coffee shelf data
FALLBACK_CONFUSION_PAIRS = [
    # ALI variants
    (6, 7, 21),    # ALI ORIGINAL filtermalt ↔ kokmalt
    (5, 7, 10),    # ALI KAFFE filtermalt ↔ ALI ORIGINAL kokmalt
    # EVERGOOD variants
    (89, 90, 20),  # EVERGOOD CLASSIC kokmalt ↔ filtermalt
    (91, 92, 12),  # EVERGOOD DARK ROAST filtermalt ↔ pressmalt
]


# ---------------------------------------------------------------------------
# Hard-Negative Sampler
# ---------------------------------------------------------------------------


class HardNegativeSampler(Sampler):
    """Balanced sampling with boosted weight for confusing categories.

    Extends BalancedConcatSampler's approach: inverse-sqrt frequency weights,
    but categories in confusion pairs get an additional boost factor.
    When a category from a pair is sampled, its counterpart is also more
    likely to appear in the same epoch.

    Args:
        shelf_labels: list of category_ids from shelf crop dataset
        ref_labels: list of category_ids from reference dataset
        confusion_pairs: list of (cat_a, cat_b, count) tuples
        boost_factor: multiplier on sampling weight for confusing categories
        shelf_ratio: fraction of epoch from shelf crops (0.7 = 70%)
        epoch_size: total samples per epoch
    """

    def __init__(
        self,
        shelf_labels,
        ref_labels,
        confusion_pairs,
        boost_factor=3.0,
        shelf_ratio=0.7,
        epoch_size=None,
    ):
        self.shelf_labels = shelf_labels
        self.ref_labels = ref_labels
        self.shelf_ratio = shelf_ratio
        self.epoch_size = epoch_size or len(shelf_labels)
        self.epoch = 0

        # Categories involved in confusion pairs
        confused_cats = set()
        for a, b, _ in confusion_pairs:
            confused_cats.add(a)
            confused_cats.add(b)

        # Compute per-class weights using inverse sqrt frequency + boost
        all_labels = shelf_labels + ref_labels
        counts = Counter(all_labels)

        def weight_for(lbl):
            w = 1.0 / math.sqrt(counts[lbl])
            if lbl in confused_cats:
                w *= boost_factor
            return w

        self.shelf_weights = torch.tensor(
            [weight_for(lbl) for lbl in shelf_labels],
            dtype=torch.float64,
        )
        self.ref_weights = (
            torch.tensor(
                [weight_for(lbl) for lbl in ref_labels],
                dtype=torch.float64,
            )
            if ref_labels
            else torch.tensor([], dtype=torch.float64)
        )

        self._len_shelf = len(shelf_labels)
        self._len_ref = len(ref_labels)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch + 42)

        n_shelf = int(self.epoch_size * self.shelf_ratio)
        n_ref = self.epoch_size - n_shelf

        indices = []

        if n_shelf > 0 and self._len_shelf > 0:
            shelf_idx = torch.multinomial(self.shelf_weights, n_shelf, replacement=True, generator=g)
            indices.append(shelf_idx)

        if n_ref > 0 and self._len_ref > 0:
            ref_idx = torch.multinomial(self.ref_weights, n_ref, replacement=True, generator=g)
            ref_idx = ref_idx + self._len_shelf
            indices.append(ref_idx)

        if indices:
            all_indices = torch.cat(indices)
            perm = torch.randperm(len(all_indices), generator=g)
            all_indices = all_indices[perm]
            return iter(all_indices.tolist())

        return iter([])

    def __len__(self):
        return self.epoch_size


# ---------------------------------------------------------------------------
# Pair-Aware Contrastive Loss
# ---------------------------------------------------------------------------


def hard_negative_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    confusion_pairs: list,
    margin: float = 0.3,
) -> torch.Tensor:
    """Push apart embeddings from confusing category pairs in the batch.

    For each pair (a, b) in confusion_pairs:
      - Find embeddings with label a and label b in the batch
      - Compute cross-similarity between them
      - Loss = mean(max(0, sim - margin)) for all cross-pairs

    Args:
        embeddings: [B, D] L2-normalized embeddings
        labels: [B] integer category labels
        confusion_pairs: list of (cat_a, cat_b, count)
        margin: target similarity ceiling for confusing pairs

    Returns:
        Scalar loss (0 if no confusion pairs present in batch).
    """
    total_loss = torch.tensor(0.0, device=embeddings.device)
    n_pairs = 0

    # L2-normalize for cosine similarity
    embeddings = F.normalize(embeddings, dim=1)

    for cat_a, cat_b, _ in confusion_pairs:
        mask_a = labels == cat_a
        mask_b = labels == cat_b
        n_a = mask_a.sum().item()
        n_b = mask_b.sum().item()

        if n_a == 0 or n_b == 0:
            continue

        embs_a = embeddings[mask_a]  # [n_a, D]
        embs_b = embeddings[mask_b]  # [n_b, D]

        # Cross-similarity: [n_a, n_b]
        cross_sim = embs_a @ embs_b.T

        # Hinge loss: push similarity below margin
        pair_loss = torch.clamp(cross_sim - margin, min=0).mean()
        total_loss = total_loss + pair_loss
        n_pairs += 1

    if n_pairs > 0:
        total_loss = total_loss / n_pairs

    return total_loss
