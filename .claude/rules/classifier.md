# Classifier Rules

## Backbone
- DINOv2 ViT-B/14, 768-dim embeddings
- Native resolution: 518×518 (14×37 patches) — use this, not 224
- Load from local weights in sandbox: `torch.load("models/dinov2_vitb14.pth")`

## Loss: Combined Distillation
$$L_{total} = L_{ArcFace} + α · L_{cosine} + β · L_{RKD-dist}$$

- **ArcFace** (pytorch-metric-learning): margin=28.6° (0.5 rad), scale=64
- **Cosine embedding loss**: 1 - cos(z_teacher, z_student) for same product, max(0, cos - margin) for different
- **RKD distance loss**: match pairwise distance structure between teacher and student mini-batch embeddings
- Default: α=0.5, β=2.0

## EMA Teacher
- NOT a frozen snapshot — an exponential moving average of the student
- Momentum cosine warmup: 0.996 → 1.0 over training
- Update after each optimizer step:
  ```python
  for t_param, s_param in zip(teacher.parameters(), student.parameters()):
      t_param.data = momentum * t_param.data + (1 - momentum) * s_param.data
  ```

## Progressive Unfreezing Schedule
1. Epochs 0-5: Freeze entire DINOv2 backbone, train ArcFace head only
2. Epochs 5-15: Unfreeze last 2 transformer blocks, lr=1e-6
3. Epochs 15+: Unfreeze last 4 transformer blocks, lr=5e-7

## Multi-GPU Training
- `SyncArcFaceLoss`: all_gather() embeddings+labels across DDP ranks before computing ArcFace
- `BalancedDistributedSampler`: oversample rare classes, 70/30 shelf-to-reference ratio
- BF16 mixed precision default
- `--grad_accum_steps` for batch size scaling (target effective batch 512)

## Transforms
- Training: `Resize(546) → RandomResizedCrop(518, scale=(0.8, 1.0))` + ColorJitter + HFlip
- Eval: `Resize(546) → CenterCrop(518) → Normalize`
- Student-teacher: teacher gets eval transform, student gets training transform

## Inference Matching
- Pre-compute reference embeddings at 518 resolution, L2-normalized
- Cosine similarity via dot product on normalized vectors
- Assign category_id of highest similarity score

## Evaluation Metrics
- Recall@1, Recall@5 against reference gallery
- Silhouette Score on embedding clusters
- UMAP visualization (not T-SNE)

## Ablation Matrix
| Parameter | Values |
|-----------|--------|
| ArcFace margin | [0.3, 0.5, 0.7] rad |
| Input resolution | [224, 448, 518] |
| Distillation α | [0.1, 0.5, 1.0] |
| Distillation β (RKD) | [1.0, 2.0, 4.0] |
| EMA momentum | [0.999 fixed, cosine 0.996→1.0] |
| Augmentation | [ColorJitter, RandomSolarize, Albumentations shelf-glare] |
