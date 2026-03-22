"""
Resolution and training configuration — single source of truth.
All datasets, transforms, and inference paths import from here.
Changing CLASSIFIER_SIZE requires re-running preprocess_crops and precompute_refs.
"""

# DINOv2 ViT-B/14: native 518 = 14 × 37 patches
CLASSIFIER_SIZE = 518
CLASSIFIER_RESIZE = 546  # short-edge resize before crop
CLASSIFIER_PATCH_SIZE = 14

# YOLO detection
DETECTOR_IMGSZ = 1280

# Crop extraction
CROP_BUFFER = 0.05  # 5% padding around bounding boxes
CROP_SAVE_SIZE = 546  # max-side for pre-extracted crop JPEGs

# ImageNet normalization (DINOv2 expected)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ArcFace defaults
EMBEDDING_DIM = 768  # DINOv2 ViT-B/14 output
NUM_CLASSES = 356
ARCFACE_MARGIN = 28.6  # degrees (~0.5 radians)
ARCFACE_SCALE = 64

# Distillation defaults
DISTILLATION_ALPHA = 0.5  # cosine embedding loss weight
DISTILLATION_BETA = 2.0  # RKD distance loss weight
EMA_MOMENTUM_START = 0.996
EMA_MOMENTUM_END = 1.0

# Training
DEFAULT_BATCH_SIZE = 128
UNFREEZE_EPOCH_STAGE1 = 5  # unfreeze last 2 blocks
UNFREEZE_EPOCH_STAGE2 = 15  # unfreeze last 4 blocks
