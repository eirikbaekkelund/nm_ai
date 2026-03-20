"""
Recall@K evaluation for the classifier.

Two eval modes:
  1. Gallery matching: cosine sim between val embeddings and reference gallery (frozen baseline)
  2. W-based accuracy: classify via ArcFace logits (embeddings @ W.T) — tracks head training

Usage:
    from vision_task.evaluate import validate
"""

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader


@torch.no_grad()
def embed_dataset(model, dataloader: DataLoader, device: torch.device):
    """Forward all batches through model, return L2-normalized embeddings + labels.

    Returns:
        (embeddings [N, 768] L2-normalized, labels [N])
    """
    all_embs = []
    all_labels = []

    model.eval()
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        with autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            embs = model(images)
        embs = F.normalize(embs.float(), dim=1)
        all_embs.append(embs.cpu())
        all_labels.extend(
            labels if isinstance(labels, list) else labels.tolist() if hasattr(labels, 'tolist') else list(labels)
        )

    return torch.cat(all_embs, dim=0), torch.tensor(all_labels, dtype=torch.long)


def aggregate_ref_embeddings(embeddings: torch.Tensor, labels: torch.Tensor):
    """Average multi-angle reference embeddings per category, re-normalize.

    Args:
        embeddings: [N_ref, 768] L2-normalized per-image embeddings
        labels: [N_ref] category_ids

    Returns:
        (ref_embs [C, 768] L2-normalized, ref_labels [C]) where C = unique categories
    """
    unique_labels = labels.unique(sorted=True)
    aggregated = []

    for lbl in unique_labels:
        mask = labels == lbl
        mean_emb = embeddings[mask].mean(dim=0)
        aggregated.append(mean_emb)

    ref_embs = torch.stack(aggregated, dim=0)
    ref_embs = F.normalize(ref_embs, dim=1)
    return ref_embs, unique_labels


def compute_recall_at_k(
    query_embs: torch.Tensor,
    query_labels: torch.Tensor,
    ref_embs: torch.Tensor,
    ref_labels: torch.Tensor,
    k_values=(1, 5),
):
    """Compute Recall@K via cosine similarity.

    Args:
        query_embs: [N_q, 768] L2-normalized
        query_labels: [N_q] category_ids
        ref_embs: [C, 768] L2-normalized per-category
        ref_labels: [C] category_ids
        k_values: tuple of K values

    Returns:
        dict mapping f"recall@{k}" -> float
    """
    # Cosine similarity: [N_q, C]
    sim = query_embs @ ref_embs.T

    results = {}
    for k in k_values:
        topk_indices = sim.topk(min(k, sim.size(1)), dim=1).indices  # [N_q, k]
        topk_labels = ref_labels[topk_indices]  # [N_q, k]
        correct = (topk_labels == query_labels.unsqueeze(1)).any(dim=1)
        results[f"recall@{k}"] = correct.float().mean().item()

    return results


@torch.no_grad()
def compute_w_accuracy(val_embs, val_labels, loss_fn, device):
    """Classify val embeddings via ArcFace W matrix (no margin, just cosine logits).

    This measures whether the ArcFace head is learning to separate classes,
    independent of the reference gallery.

    Args:
        val_embs: [N, 768] L2-normalized
        val_labels: [N] category_ids
        loss_fn: ArcFaceLoss with trained W matrix
        device: torch device

    Returns:
        dict with w_acc_top1, w_acc_top5
    """
    # ArcFace W shape: [embedding_dim, num_classes] in pytorch-metric-learning
    W = loss_fn.W  # [768, 356]
    W_norm = F.normalize(W, dim=0)  # L2-normalize each class column

    # Cosine logits: [N, num_classes]
    logits = val_embs.to(device) @ W_norm

    results = {}
    for k, name in [(1, "w_acc_top1"), (5, "w_acc_top5")]:
        topk_preds = logits.topk(k, dim=1).indices  # [N, k]
        correct = (topk_preds == val_labels.to(device).unsqueeze(1)).any(dim=1)
        results[name] = correct.float().mean().item()

    return results


@torch.no_grad()
def validate(model, val_loader: DataLoader, ref_loader: DataLoader, device: torch.device, loss_fn=None):
    """Full validation pipeline.

    Reports both gallery-matching recall (frozen baseline) and W-based accuracy
    (tracks ArcFace head training).

    Args:
        model: GroceryEmbedder
        val_loader: validation shelf crops
        ref_loader: reference product images (eval transform)
        device: torch device
        loss_fn: optional ArcFaceLoss — if provided, also computes W-based accuracy

    Returns:
        dict with recall@1, recall@5, w_acc_top1, w_acc_top5, n_val, n_ref_categories
    """
    model.eval()

    # Embed validation crops
    val_embs, val_labels = embed_dataset(model, val_loader, device)

    # Embed reference images and aggregate per-category
    ref_embs_raw, ref_labels_raw = embed_dataset(model, ref_loader, device)
    ref_embs, ref_labels = aggregate_ref_embeddings(ref_embs_raw, ref_labels_raw)

    # Gallery-matching recall (frozen DINOv2 baseline)
    metrics = compute_recall_at_k(val_embs, val_labels, ref_embs, ref_labels, k_values=(1, 5))
    metrics["n_val"] = len(val_labels)
    metrics["n_ref_categories"] = len(ref_labels)

    # W-based accuracy (tracks ArcFace head learning)
    if loss_fn is not None:
        w_metrics = compute_w_accuracy(val_embs, val_labels, loss_fn, device)
        metrics.update(w_metrics)

    return metrics
