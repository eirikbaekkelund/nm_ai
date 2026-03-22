"""
Single source of truth for all transform pipelines.
All sizes imported from config.py — change resolution there, not here.
"""

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from vision_task.config import (
    CLASSIFIER_RESIZE,
    CLASSIFIER_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def get_train_transform():
    """Student training transform with augmentation."""
    return T.Compose(
        [
            T.Resize(CLASSIFIER_RESIZE, interpolation=InterpolationMode.BICUBIC),
            T.RandomResizedCrop(CLASSIFIER_SIZE, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transform():
    """Deterministic eval/inference transform."""
    return T.Compose(
        [
            T.Resize(CLASSIFIER_RESIZE, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(CLASSIFIER_SIZE),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_teacher_transform():
    """Teacher sees clean images — same as eval."""
    return get_eval_transform()
