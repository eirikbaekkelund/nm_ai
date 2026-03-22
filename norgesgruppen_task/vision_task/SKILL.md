# DINOv2 Grocery Research Setup

## Strategy: Dense Matching via Metric Learning
1. **Stage 1:** YOLO11x (Class-Agnostic) to find all "products".
2. **Stage 2:** DINOv2 (ViT-B/14) Backbone + ArcFace Head for product ID.

## Research Phases for Claude
- **Phase 1: Synthetic Data Generation.** Use `rembg` on reference images and paste them onto Norwegian shelf backgrounds with random lighting/blur.
- **Phase 2: Linear Probing.** Freeze DINOv2. Train only the ArcFace head to cluster the 356 classes.
- **Phase 3: Fine-Tuning.** Unfreeze the last 4 blocks of DINOv2 using a very low learning rate ($1e-6$).
- **Phase 4: Knowledge Distillation.** (Student-Teacher) Use the frozen DINOv2 as a teacher to guide a smaller ViT student if VRAM becomes an issue (though L4 handles ViT-B fine).

## Ablation Study Goals
- Margin $m$ in ArcFace: [0.3, 0.5, 0.7]
- Input Resolution: [224, 448, 518] (DINOv2 performs significantly better at native 518 for small details).
- Augmentation Impact: Test 'ColorJitter' vs 'RandomSolarize' (plastic glare simulation).