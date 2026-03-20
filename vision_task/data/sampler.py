"""
BalancedConcatSampler — weighted sampling across shelf crops + reference images.

Single-GPU version. DDP extension deferred to Phase 5.

Usage:
    from vision_task.data.sampler import BalancedConcatSampler
"""

import math
from collections import Counter

import torch
from torch.utils.data import Sampler


class BalancedConcatSampler(Sampler):
    """Balanced sampling across a ConcatDataset of [shelf_crops, reference_images].

    - Inverse-sqrt-frequency class weights (softer than 1/count)
    - Configurable shelf-to-reference ratio per epoch
    - set_epoch() for reproducible shuffling
    """

    def __init__(self, shelf_labels, ref_labels, shelf_ratio=0.7, epoch_size=None):
        """
        Args:
            shelf_labels: list of category_ids from shelf crop dataset
            ref_labels: list of category_ids from reference dataset
            shelf_ratio: fraction of each epoch drawn from shelf crops (0.7 = 70%)
            epoch_size: total samples per epoch; defaults to len(shelf_labels)
        """
        self.shelf_labels = shelf_labels
        self.ref_labels = ref_labels
        self.shelf_ratio = shelf_ratio
        self.epoch_size = epoch_size or len(shelf_labels)
        self.epoch = 0

        # Compute per-class weights using inverse sqrt frequency
        all_labels = shelf_labels + ref_labels
        counts = Counter(all_labels)

        # Shelf sample weights: indices [0, len_shelf)
        self.shelf_weights = torch.tensor(
            [1.0 / math.sqrt(counts[lbl]) for lbl in shelf_labels],
            dtype=torch.float64,
        )

        # Ref sample weights: indices [len_shelf, len_shelf + len_ref)
        self.ref_weights = (
            torch.tensor(
                [1.0 / math.sqrt(counts[lbl]) for lbl in ref_labels],
                dtype=torch.float64,
            )
            if ref_labels
            else torch.tensor([], dtype=torch.float64)
        )

        self._len_shelf = len(shelf_labels)
        self._len_ref = len(ref_labels)

    def set_epoch(self, epoch):
        """For reproducible shuffling across epochs."""
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch + 42)

        n_shelf = int(self.epoch_size * self.shelf_ratio)
        n_ref = self.epoch_size - n_shelf

        indices = []

        # Sample from shelf crops
        if n_shelf > 0 and self._len_shelf > 0:
            shelf_idx = torch.multinomial(self.shelf_weights, n_shelf, replacement=True, generator=g)
            indices.append(shelf_idx)

        # Sample from reference images (offset by shelf length)
        if n_ref > 0 and self._len_ref > 0:
            ref_idx = torch.multinomial(self.ref_weights, n_ref, replacement=True, generator=g)
            ref_idx = ref_idx + self._len_shelf  # offset into ConcatDataset
            indices.append(ref_idx)

        if indices:
            all_indices = torch.cat(indices)
            # Shuffle the interleaved indices
            perm = torch.randperm(len(all_indices), generator=g)
            all_indices = all_indices[perm]
            return iter(all_indices.tolist())

        return iter([])

    def __len__(self):
        return self.epoch_size
