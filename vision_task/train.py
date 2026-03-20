"""
Phase 4: ArcFace linear probe — freeze DINOv2, train ArcFace head only.

Usage:
    python -m vision_task.train [--epochs 20] [--batch_size 128] [--lr 1e-3]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from vision_task.config import EMBEDDING_DIM, NUM_CLASSES
from vision_task.data.dataloader import create_dataloaders
from vision_task.data.reference_dataset import ProductReferenceDataset
from vision_task.data.transforms import get_eval_transform
from vision_task.embedder import GroceryEmbedder
from vision_task.evaluate import validate
from vision_task.losses import create_arcface_loss

logger = logging.getLogger(__name__)


def _auto_batch_size() -> int:
    """Pick batch size based on available GPU VRAM."""
    if not torch.cuda.is_available():
        return 16
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if vram_gb >= 40:  # H100 / A100
        return 128
    if vram_gb >= 20:  # L4 / A5000
        return 64
    if vram_gb >= 10:  # RTX 3080 / etc
        return 32
    return 16  # 8 GB dev GPUs


def _auto_num_workers() -> int:
    """Conservative workers: 4 on Windows (spawn), min(8, cores-2) on Linux."""
    import os
    import platform

    cores = os.cpu_count() or 4
    if platform.system() == "Windows":
        return min(8, cores)
    return min(8, max(1, cores - 2))


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4: ArcFace linear probe")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=None, help="Auto-detected from VRAM if not set")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_every", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default="experiments/phase4_linear_probe")
    parser.add_argument("--num_workers", type=int, default=None, help="Auto-detected from OS/cores if not set")
    parser.add_argument("--weights_path", type=str, default="models/dinov2_vitb14.pth")
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = _auto_batch_size()
    if args.num_workers is None:
        args.num_workers = _auto_num_workers()
    return args


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Log to both console and file
    log_file = save_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
    )
    # JSON metrics log — one line per event, easy to parse
    metrics_file = save_dir / "metrics.jsonl"
    metrics_fh = open(metrics_file, "w", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info("Device: %s (%s, %.1f GB)", device, torch.cuda.get_device_name(0), vram_gb)
    else:
        logger.info("Device: %s", device)
    logger.info("Config: batch_size=%d, num_workers=%d, lr=%.1e", args.batch_size, args.num_workers, args.lr)

    # --- Model ---
    model = GroceryEmbedder(
        weights_path=args.weights_path,
        freeze_backbone=True,
    ).to(device)

    n_params_total = sum(p.numel() for p in model.parameters())
    n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Backbone params: %d total, %d trainable (should be 0)", n_params_total, n_params_trainable)

    # --- Loss (trainable ArcFace head) ---
    loss_fn = create_arcface_loss(
        num_classes=NUM_CLASSES,
        embedding_dim=EMBEDDING_DIM,
    ).to(device)

    n_loss_params = sum(p.numel() for p in loss_fn.parameters())
    logger.info("ArcFace head params: %d", n_loss_params)

    # --- Optimizer: only ArcFace W matrix ---
    optimizer = torch.optim.AdamW(loss_fn.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Data ---
    train_loader, val_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    logger.info("Train batches: %d, Val batches: %d", len(train_loader), len(val_loader))

    # Reference images with eval transform (for validation gallery)
    ref_val_ds = ProductReferenceDataset(
        product_images_dir="data/product_images",
        mapping_path="data/category_mapping.json",
        transform=get_eval_transform(),
    )
    ref_val_loader = DataLoader(
        ref_val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logger.info("Reference images: %d across %d products", len(ref_val_ds), len(set(ref_val_ds.labels)))

    # Log full config to JSON
    config_record = {
        "event": "config",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "num_workers": args.num_workers,
        "val_every": args.val_every,
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "n_ref_images": len(ref_val_ds),
        "n_ref_products": len(set(ref_val_ds.labels)),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    metrics_fh.write(json.dumps(config_record) + "\n")
    metrics_fh.flush()

    # --- Training loop ---
    best_recall1 = 0.0
    use_amp = device.type == "cuda"

    for epoch in range(args.epochs):
        model.train()  # frozen backbone stays in eval via override
        loss_fn.train()

        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = torch.as_tensor(labels, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)

            # BF16 for backbone forward, FP32 for ArcFace
            if use_amp:
                with autocast("cuda", dtype=torch.bfloat16):
                    embeddings = model(images)
                embeddings = embeddings.float()
            else:
                embeddings = model(images).float()

            loss = loss_fn(embeddings, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        logger.info(
            "Epoch %d/%d — loss: %.4f, lr: %.2e, time: %.1fs",
            epoch + 1,
            args.epochs,
            avg_loss,
            lr_now,
            elapsed,
        )
        metrics_fh.write(
            json.dumps(
                {
                    "event": "epoch",
                    "epoch": epoch + 1,
                    "train_loss": round(avg_loss, 4),
                    "lr": lr_now,
                    "time_s": round(elapsed, 1),
                }
            )
            + "\n"
        )
        metrics_fh.flush()

        # --- Validation ---
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            metrics = validate(model, val_loader, ref_val_loader, device, loss_fn=loss_fn)
            w_top1 = metrics.get("w_acc_top1", 0)
            w_top5 = metrics.get("w_acc_top5", 0)
            logger.info(
                "  Val: R@1=%.4f R@5=%.4f | W_acc@1=%.4f W_acc@5=%.4f (n_val=%d, n_ref=%d)",
                metrics["recall@1"],
                metrics["recall@5"],
                w_top1,
                w_top5,
                metrics["n_val"],
                metrics["n_ref_categories"],
            )
            metrics_fh.write(
                json.dumps(
                    {
                        "event": "val",
                        "epoch": epoch + 1,
                        **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
                    }
                )
                + "\n"
            )
            metrics_fh.flush()

            # Save best checkpoint
            if metrics["recall@1"] > best_recall1:
                best_recall1 = metrics["recall@1"]
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "loss_fn_state_dict": loss_fn.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                }
                best_path = save_dir / "best.pt"
                torch.save(checkpoint, best_path)
                logger.info("  Saved best checkpoint: R@1=%.4f -> %s", best_recall1, best_path)

    # Save final checkpoint
    final_checkpoint = {
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "loss_fn_state_dict": loss_fn.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": {"final_train_loss": avg_loss},
    }
    torch.save(final_checkpoint, save_dir / "final.pt")
    logger.info("Training complete. Best R@1: %.4f", best_recall1)
    metrics_fh.write(
        json.dumps(
            {
                "event": "done",
                "best_recall1": round(best_recall1, 4),
                "final_train_loss": round(avg_loss, 4),
            }
        )
        + "\n"
    )
    metrics_fh.close()


if __name__ == "__main__":
    main()
