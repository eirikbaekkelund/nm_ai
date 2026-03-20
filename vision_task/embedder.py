"""
DINOv2 ViT-B/14 embedder via timm.

- Dev: falls back to pretrained=True download if local weights missing
- Sandbox: loads from local .pth file (no network)
- Forward returns raw CLS token [B, 768] — NO L2 normalization
"""

import logging
from pathlib import Path

import timm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_TIMM_MODEL_NAME = "vit_base_patch14_dinov2.lvd142m"


class GroceryEmbedder(nn.Module):
    def __init__(
        self,
        weights_path: str | None = "models/dinov2_vitb14.pth",
        freeze_backbone: bool = True,
    ):
        super().__init__()

        if weights_path is None:
            # Architecture only — caller will load checkpoint via load_state_dict
            logger.info("Creating DINOv2 architecture (no weights)")
            self.backbone = timm.create_model(
                _TIMM_MODEL_NAME,
                pretrained=False,
                num_classes=0,
            )
        else:
            weights_file = Path(weights_path)
            if weights_file.exists():
                logger.info("Loading DINOv2 from local weights: %s", weights_path)
                self.backbone = timm.create_model(
                    _TIMM_MODEL_NAME,
                    pretrained=False,
                    num_classes=0,
                )
                state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)
                self.backbone.load_state_dict(state_dict, strict=True)
            else:
                logger.info("Local weights not found at %s — downloading via timm", weights_path)
                self.backbone = timm.create_model(
                    _TIMM_MODEL_NAME,
                    pretrained=True,
                    num_classes=0,
                )

        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 3, 518, 518] -> [B, 768] raw CLS token (no L2 norm)."""
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def unfreeze_last_n_blocks(self, n: int):
        """Unfreeze the last n transformer blocks for fine-tuning."""
        for block in self.backbone.blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
            block.train()

    def train(self, mode: bool = True):
        """Override train() to keep frozen layers in eval mode."""
        super().train(mode)
        # If backbone is frozen, keep it in eval mode (stable BN/dropout)
        if not any(p.requires_grad for p in self.backbone.blocks[0].parameters()):
            self.backbone.eval()
        return self
