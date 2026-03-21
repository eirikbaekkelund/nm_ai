"""
Pre-compute L2-normalized reference embeddings for sandbox inference.

Loads trained DINOv2 from checkpoint, embeds all reference product images,
averages per-category, and saves to models/ref_embeddings.pt.

Accepts both formats:
  - Training checkpoint: dict with "model_state_dict" key
  - Stripped state_dict: raw {layer_name: tensor} (e.g. models/classifier.pt)

TTA mode (--tta): embeds each reference image with multiple augmented views
(original, horizontal flip, 4-corner crops), averages all views per category.

Usage:
    python -m vision_task.precompute_refs --checkpoint models/classifier.pt
    python -m vision_task.precompute_refs --checkpoint models/classifier.pt --tta
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms as T

from vision_task.config import CLASSIFIER_RESIZE, CLASSIFIER_SIZE, IMAGENET_MEAN, IMAGENET_STD
from vision_task.data.reference_dataset import ProductReferenceDataset
from vision_task.data.transforms import get_eval_transform
from vision_task.embedder import GroceryEmbedder
from vision_task.evaluate import embed_dataset


logger = logging.getLogger(__name__)


class TTAReferenceDataset(Dataset):
    """Wraps a ProductReferenceDataset with multiple augmented views per image.

    For each image, produces N views:
      0: center crop (standard eval)
      1: horizontal flip + center crop
      2-5: four corner crops (TL, TR, BL, BR)

    Each view gets its own entry; labels are duplicated accordingly.
    """

    def __init__(self, base_dataset: ProductReferenceDataset):
        self._base = base_dataset
        self._n_views = 6  # center, hflip, TL, TR, BL, BR

        # Shared resize + normalize
        self._resize = T.Resize(CLASSIFIER_RESIZE, interpolation=InterpolationMode.BICUBIC)
        self._normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self._to_tensor = T.ToTensor()

    def __len__(self):
        return len(self._base) * self._n_views

    def __getitem__(self, idx):
        base_idx = idx // self._n_views
        view_idx = idx % self._n_views

        entry = self._base._entries[base_idx]
        from PIL import Image

        img = Image.open(entry["path"]).convert("RGB")

        # Resize to CLASSIFIER_RESIZE on short edge
        img = self._resize(img)
        w, h = img.size
        cs = CLASSIFIER_SIZE

        if view_idx == 0:
            # Center crop (standard eval)
            img = T.functional.center_crop(img, [cs, cs])
        elif view_idx == 1:
            # Horizontal flip + center crop
            img = T.functional.hflip(img)
            img = T.functional.center_crop(img, [cs, cs])
        elif view_idx == 2:
            # Top-left
            img = T.functional.crop(img, 0, 0, cs, cs)
        elif view_idx == 3:
            # Top-right
            img = T.functional.crop(img, 0, max(0, w - cs), cs, cs)
        elif view_idx == 4:
            # Bottom-left
            img = T.functional.crop(img, max(0, h - cs), 0, cs, cs)
        elif view_idx == 5:
            # Bottom-right
            img = T.functional.crop(img, max(0, h - cs), max(0, w - cs), cs, cs)

        tensor = self._to_tensor(img)
        tensor = self._normalize(tensor)
        return tensor, entry["category_id"]

    @property
    def labels(self):
        return [lbl for lbl in self._base._labels for _ in range(self._n_views)]


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-compute reference embeddings")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/classifier.pt",
        help="Path to classifier weights (.pt) — training checkpoint or stripped state_dict",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/ref_embeddings.pt",
        help="Output path for reference embeddings",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--product_images_dir",
        type=str,
        default="data/product_images",
    )
    parser.add_argument(
        "--mapping_path",
        type=str,
        default="data/category_mapping.json",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Use TTA: embed each reference with 6 views (center, hflip, 4 corners), average per category",
    )
    parser.add_argument(
        "--angles",
        type=str,
        default="main,front",
        help="Comma-separated reference image angles to use (e.g. 'main,front'). 'all' = use every angle.",
    )
    return parser.parse_args()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- Load model from checkpoint ---
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Auto-detect format: training checkpoint (has "model_state_dict" key) vs raw state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        if "metrics" in checkpoint:
            logger.info("Checkpoint metrics: %s", checkpoint["metrics"])
    else:
        state_dict = checkpoint

    model = GroceryEmbedder(weights_path=None, freeze_backbone=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # --- Load reference dataset ---
    angles = None if args.angles == "all" else args.angles.split(",")
    logger.info("Reference angles filter: %s", angles or "all")

    base_ref_ds = ProductReferenceDataset(
        product_images_dir=args.product_images_dir,
        mapping_path=args.mapping_path,
        transform=None if args.tta else get_eval_transform(),
        angles=angles,
    )

    if args.tta:
        ref_ds = TTAReferenceDataset(base_ref_ds)
        logger.info(
            "TTA mode: %d base images x 6 views = %d embeddings",
            len(base_ref_ds),
            len(ref_ds),
        )
    else:
        ref_ds = base_ref_ds

    ref_loader = DataLoader(
        ref_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logger.info("Reference images: %d across %d products", len(ref_ds), len(set(ref_ds.labels)))

    # --- Embed ---
    ref_embs_raw, ref_labels_raw = embed_dataset(model, ref_loader, device)
    ref_embs_raw = F.normalize(ref_embs_raw.float(), dim=1)
    logger.info("Raw reference embeddings: %s", ref_embs_raw.shape)

    unique_cats = ref_labels_raw.unique()
    logger.info(
        "Keeping all %d individual embeddings across %d categories (max-sim matching)",
        ref_embs_raw.shape[0],
        len(unique_cats),
    )

    # --- Save ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "embeddings": ref_embs_raw,  # [N_ref, 768] L2-normalized, one per image
            "category_ids": ref_labels_raw,  # [N_ref] int64
        },
        output_path,
    )

    logger.info("Saved reference embeddings to %s", output_path)
    logger.info("Shape: %s, dtype: %s", ref_embs_raw.shape, ref_embs_raw.dtype)

    # Verify normalization
    norms = ref_embs_raw.norm(dim=1)
    logger.info("Norm range: [%.6f, %.6f] (should be ~1.0)", norms.min(), norms.max())


if __name__ == "__main__":
    main()
