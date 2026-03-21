"""
Context-Aware Query Expansion (CAQE) — inference-time spatial context boost.

Products near each other on shelves share context (same category section).
After embedding all crops for an image, expand each embedding with its
K nearest spatial neighbors' embeddings.

Usage:
    from vision_task.caqe import apply_caqe
    expanded = apply_caqe(embeddings, boxes_xyxy, k=3, alpha=0.5)
"""

import torch
import torch.nn.functional as F


def apply_caqe(
    embeddings: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    k: int = 3,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Expand each crop embedding with its K nearest spatial neighbors.

    Args:
        embeddings: [N, D] L2-normalized crop embeddings for ONE image.
        boxes_xyxy: [N, 4] detection boxes (x1, y1, x2, y2).
        k: number of spatial neighbors to aggregate.
        alpha: weight on original embedding (1-alpha on neighbor mean).

    Returns:
        [N, D] L2-normalized context-expanded embeddings.
    """
    n = embeddings.shape[0]
    if k <= 0 or n <= 1 or alpha >= 1.0:
        return embeddings

    # Effective k: can't have more neighbors than n-1
    eff_k = min(k, n - 1)

    # Box centroids: [N, 2]
    centroids = torch.stack([
        (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2,
        (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2,
    ], dim=1)

    # Pairwise L2 distances: [N, N]
    dists = torch.cdist(centroids.unsqueeze(0).float(), centroids.unsqueeze(0).float()).squeeze(0)

    # Set self-distance to inf so we don't pick self as neighbor
    dists.fill_diagonal_(float("inf"))

    # Find k nearest neighbors per crop
    _, nn_indices = dists.topk(eff_k, dim=1, largest=False)  # [N, eff_k]

    # Gather neighbor embeddings and average
    neighbor_embs = embeddings[nn_indices]  # [N, eff_k, D]
    neighbor_mean = neighbor_embs.mean(dim=1)  # [N, D]

    # Weighted combination
    expanded = alpha * embeddings + (1 - alpha) * neighbor_mean

    # Re-normalize
    return F.normalize(expanded, dim=1)
