"""Dataloader factory for training and evaluation.

Usage:
    from vision_task.data.dataloader import create_dataloaders
"""

from torch.utils.data import ConcatDataset, DataLoader

from vision_task.config import DEFAULT_BATCH_SIZE
from vision_task.data.datasets import (
    PreExtractedCropDataset,
    TransformWrapper,
    get_train_val_indices,
)
from vision_task.data.reference_dataset import ProductReferenceDataset
from vision_task.data.sampler import BalancedConcatSampler
from vision_task.data.transforms import get_eval_transform, get_train_transform


def create_dataloaders(
    crops_dir="data/crops",
    manifest_path="data/crops/crops_manifest.csv",
    product_images_dir="data/product_images",
    mapping_path="data/category_mapping.json",
    batch_size=DEFAULT_BATCH_SIZE,
    num_workers=8,
    exclude_unknown=True,
):
    """Create training and validation dataloaders.

    Train: shelf crops (train split) + reference images, balanced sampler, drop_last=True
    Val: shelf crops (val split) only, sequential, no sampling

    Returns:
        (train_loader, val_loader)
    """
    train_transform = get_train_transform()
    eval_transform = get_eval_transform()

    # Load full shelf crop dataset (no transform — applied per-split via wrapper)
    shelf_ds = PreExtractedCropDataset(
        crops_dir=crops_dir,
        manifest_path=manifest_path,
        transform=None,
        exclude_unknown=exclude_unknown,
    )

    # Image-level train/val split matching YOLO
    train_indices, val_indices = get_train_val_indices(shelf_ds)

    # Wrap splits with appropriate transforms
    shelf_train = TransformWrapper(shelf_ds, train_indices, train_transform)
    shelf_val = TransformWrapper(shelf_ds, val_indices, eval_transform)

    # Reference images with train transform
    ref_ds = ProductReferenceDataset(
        product_images_dir=product_images_dir,
        mapping_path=mapping_path,
        transform=train_transform,
    )

    # Concatenate shelf train + reference for training
    train_ds = ConcatDataset([shelf_train, ref_ds])

    # Balanced sampler
    sampler = BalancedConcatSampler(
        shelf_labels=shelf_train.labels,
        ref_labels=ref_ds.labels,
        shelf_ratio=0.7,
        epoch_size=len(shelf_train),
    )

    worker_kwargs = {}
    if num_workers > 0:
        worker_kwargs["persistent_workers"] = True
        worker_kwargs["prefetch_factor"] = 2 if num_workers <= 4 else 4

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        **worker_kwargs,
    )

    val_loader = DataLoader(
        shelf_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        **worker_kwargs,
    )

    return train_loader, val_loader
