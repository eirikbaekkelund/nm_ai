"""
ArcFace loss factory for Phase 4 linear probe.

Phase 5 will add CombinedLoss (ArcFace + cosine + RKD) and EMA teacher here.
"""

from pytorch_metric_learning.losses import ArcFaceLoss

from vision_task.config import ARCFACE_MARGIN, ARCFACE_SCALE, EMBEDDING_DIM, NUM_CLASSES


def create_arcface_loss(
    num_classes: int = NUM_CLASSES,
    embedding_dim: int = EMBEDDING_DIM,
    margin: float = ARCFACE_MARGIN,
    scale: float = ARCFACE_SCALE,
) -> ArcFaceLoss:
    """Create ArcFace loss with trainable weight matrix W [embedding_dim, num_classes].

    Args:
        num_classes: number of product categories (356)
        embedding_dim: DINOv2 output dimension (768)
        margin: angular margin in degrees (28.6 ~ 0.5 rad)
        scale: cosine scaling factor (64)
    """
    return ArcFaceLoss(
        num_classes=num_classes,
        embedding_size=embedding_dim,
        margin=margin,
        scale=scale,
    )
