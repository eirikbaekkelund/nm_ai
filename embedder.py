"""Minimal inference-only DINOv2 embedder for sandbox submission.

Uses timm 0.9.12 (pre-installed in sandbox) to create DINOv2 ViT-B/14.
Forward returns raw CLS token [B, 768] — no L2 normalization.
"""

import timm
import torch
import torch.nn as nn


class GroceryEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=False,
            num_classes=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
