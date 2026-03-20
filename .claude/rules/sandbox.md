# Sandbox Inference Rules

## Hardware
- NVIDIA L4, 24 GB VRAM
- No network access — all weights must be bundled in .zip
- Sandboxed Docker container

## Weight Loading
- DINOv2: `torch.load("models/dinov2_vitb14.pth")` — NEVER `torch.hub.load()`
- YOLO: `YOLO("models/yolo_best.pt")`
- Reference embeddings: `torch.load("models/ref_embeddings.pt")` — shape [327, 768], L2-normalized

## Inference Pipeline (run.py)
1. Load shelf image
2. YOLO detect → class-agnostic bounding boxes, filter by confidence
3. Crop boxes from original image
4. Apply eval transform: `Resize(546) → CenterCrop(518) → Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])`
5. Batch crops (batch_size=64) → DINOv2 → L2-normalize embeddings
6. Cosine similarity: `torch.mm(query, ref_embeddings.T)`
7. Assign category_id of highest similarity
8. Output predictions in competition format

## VRAM Budget
| Component | VRAM |
|-----------|------|
| YOLO11x at 1280 | ~4 GB |
| DINOv2 ViT-B/14 at 518 | ~2 GB |
| Batch-64 crops at 518×518 | ~6 GB |
| Reference embeddings (327×768) | ~1 MB |
| **Total** | **~12 GB** |
| L4 headroom | 12 GB remaining ✅ |

## Safety
- Keep total under 20 GB (4 GB headroom)
- Use `torch.cuda.empty_cache()` between detection and classification if OOM
- Profile with `torch.cuda.max_memory_allocated()` during testing
- Use `@torch.no_grad()` for all inference
- Category 356 = `unknown_product` — use confidence threshold on cosine similarity
