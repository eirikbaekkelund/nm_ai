# Research Scratchpad — Phase 8 Marginal Gains

> Updated: 2026-03-21
> Status: Collecting diagnostics + planning next experiments

## Current Baselines

| Component | Metric | Value |
|-----------|--------|-------|
| Detection (YOLO11x D1) | mAP@50 | 0.934 |
| Detection (YOLO11x D1) | Recall | 0.905 |
| Classification (Phase 5) | R@1 | 0.858 |
| Classification (Phase 5) | W_acc@1 | 0.899 |
| **Combined (est.)** | **0.7×det + 0.3×cls** | **~0.91** |

---

## 1. Failure Diagnosis (TODO — run on RunPod)

### Run command
```bash
cd ~/nm_ai && git pull
python -m vision_task.diagnose \
    --classifier_weights experiments/phase5b_unfreeze4/best.pt \
    --output_dir experiments/diagnostics
```

### What we need to learn

**Detection:**
- [ ] Which size bucket has worst recall? (tiny/small/medium/large)
- [ ] Which specific images have the most missed detections?
- [ ] Are FPs clustered in certain image regions (e.g., shelf edges, price tags)?
- [ ] How many FNs are at the image boundary vs center?

**Classification:**
- [ ] What's the accuracy on matched (IoU>0.5) detections?
- [ ] Which category pairs are most confused?
- [ ] What's the cosine similarity distribution for correct vs incorrect?
- [ ] Are errors correlated with box size? (small crops → worse classification)
- [ ] Categories with 0% accuracy — do they have reference images?
- [ ] `unknown_product` (cat 355) — how many GT instances, how many predicted?

---

## 2. Detection Improvement Research

### 2.1 SKU-110K Pretraining (D2) — HIGH PRIORITY

**Paper:** Goldman et al., "Precise Detection in Densely Packed Scenes", CVPR 2019

| Property | Value |
|----------|-------|
| Images | 11,762 (train 8,219 / val 588 / test 2,936) |
| Annotations | ~1.73M bounding boxes |
| Avg objects/image | ~147 |
| Download | `http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz` |
| Size | ~13.6 GB compressed |
| Format | CSV: `image_name, x1, y1, x2, y2, class, image_width, image_height` |

**Expected gain:** +2-5% mAP from domain-specific pretraining. The dataset contains thousands of shelf images from supermarkets across US, Europe, East Asia — excellent domain match.

**Published SKU-110K results:**
- YOLO-RACE (YOLOv8 + attention): mAP@50 = 0.902
- DBA-YOLO (YOLOv10-based): ~0.91
- DETR-R50 (isalia99): mAP = 0.589

**Training plan:**
```bash
# Phase 1: Pretrain on SKU-110K (50 epochs)
yolo detect train model=yolo11x.pt data=vision_task/detection/sku110k.yaml \
    epochs=50 imgsz=1280 batch=16 lr0=0.01 mosaic=1.0 mixup=0.3

# Phase 2: Fine-tune on shelf data (30 epochs, lower LR)
yolo detect train model=runs/detect/sku110k/weights/best.pt \
    data=vision_task/detection/shelf.yaml \
    epochs=30 imgsz=1280 batch=8 lr0=0.001
```

**Blocker:** 13.6 GB download + ~12 GB uncompressed = ~26 GB. RunPod overlay is only 20 GB. Need either:
- RunPod with larger volume (50+ GB)
- Process only train split
- Use a subset (random 50%)

**Our existing code:** `vision_task/detection/convert_sku110k.py` and `sku110k.yaml` are already written and ready.

Sources:
- [SKU-110K GitHub](https://github.com/eg4000/SKU110K_CVPR19)
- [SKU-110K on Kaggle](https://www.kaggle.com/datasets/thedatasith/sku110k-annotations)
- [Ultralytics SKU-110K Docs](https://docs.ultralytics.com/datasets/detect/sku-110k/)

---

### 2.2 DETR-R50 SKU-110K (D3) — MEDIUM PRIORITY

**Model:** [`isalia99/detr-resnet-50-sku110k`](https://huggingface.co/isalia99/detr-resnet-50-sku110k)

| Property | Value |
|----------|-------|
| Architecture | DETR + ResNet-50, num_queries=400 |
| Parameters | 41.7M |
| mAP on SKU-110K val | 58.9 |
| Size FP16 | ~83 MB |
| Training code | [Isalia20/DETR-finetune](https://github.com/Isalia20/DETR-finetune) |

**Key advantage:** NMS-free. Set prediction avoids the failure mode where NMS kills overlapping true detections on tightly packed shelves.

**Fine-tuning plan on 248 images:**
1. Freeze ResNet-50 backbone, train decoder + cls head (20-30 epochs, lr=1e-5)
2. Unfreeze backbone (30-50 epochs, lr_backbone=1e-6)
3. Disable auxiliary losses (`--no_aux_loss`) to reduce overfitting

**Sandbox constraint:** `transformers` is NOT installed. Must either:
- Export to ONNX (preferred — `onnxruntime-gpu` is available)
- Include model class in .py files + load state_dict

**Primary value:** Ensemble diversity. DETR alone may not beat YOLO, but YOLO + DETR via WBF should exceed either alone.

Sources:
- [HuggingFace Model](https://huggingface.co/isalia99/detr-resnet-50-sku110k)
- [DETR Fine-tuning Guide](https://huggingface.co/learn/cookbook/en/fine_tuning_detr_custom_dataset)

---

### 2.3 Weighted Box Fusion (WBF) Ensemble — HIGH PRIORITY

**Paper:** Solovyev et al., 2021

**Method:** Fuse all predicted boxes using confidence-weighted averaging. Never discards boxes (unlike NMS). Pre-installed in sandbox: `ensemble-boxes 1.0.9`.

**Best setup:** Architecturally diverse models (YOLO + DETR). Even multi-scale YOLO with WBF gives +1-2% mAP.

**Implementation (already in ablation_sweep.py):**
```python
from ensemble_boxes import weighted_boxes_fusion
boxes, scores, labels = weighted_boxes_fusion(
    boxes_list, scores_list, labels_list,
    iou_thr=0.5, skip_box_thr=0.01
)
```

**Competition track record:** Top-3 on COCO leaderboard, winner of Waymo & Lyft detection challenges.

Sources:
- [WBF Paper](https://arxiv.org/abs/1910.13302)
- [WBF GitHub](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)

---

### 2.4 YOLO26 with STAL — LOW PRIORITY (already training)

YOLO26 introduces Small-Target-Aware Label Assignment (STAL) which specifically improves small object detection. Products on 1280px shelf images are small.

Already have YOLO26x trained. Compare to YOLO11x in ablation sweep.

---

### 2.5 Multi-Scale + TTA Detection — MEDIUM PRIORITY

- Horizontal flip + scales [1024, 1280, 1536], fuse with WBF
- Expected: +1-2% mAP on small objects
- Already implemented in `ablation_sweep.py` — need results

---

### 2.6 NMS IoU Tuning — LOW PRIORITY (ablation sweep)

For densely packed shelves, higher NMS IoU (0.8-0.9) keeps more overlapping true detections. Default is 0.7. Sweep in ablation.

---

## 3. Classification Improvement Research

### 3.1 Context-Aware Query Expansion (CAQE) — HIGH PRIORITY, FREE LUNCH

**Paper:** "Context-Aware Fine-Grained Product Recognition", IEEE Access 2025

**Core idea:** At inference time, expand each product's embedding with its K spatially nearest shelf neighbors' embeddings. Products on shelves tend to be grouped by category (all coffees together, all cereals together). Leveraging this spatial context improves matching.

**Method:**
1. Detect all products on shelf image
2. Embed all crops
3. For each crop embedding, find K=3 spatially nearest neighbors (by bbox centroid distance)
4. Average the crop's embedding with its neighbors' embeddings (weighted by spatial proximity)
5. L2-normalize the expanded embedding
6. Match to reference gallery

**Expected gain:** +1-3% classification accuracy. Zero retraining cost.

**Practical notes:**
- K=3 was optimal in the paper
- Weight spatial neighbors by 1/distance or exponential decay
- Only average within the same image (not cross-image)
- The intuition: if 3 products in a row are all classified as coffee brand X, but the middle one is uncertain, the context pushes it toward coffee

**Implementation plan:** Add to `run.py` after embedding all crops for an image, before cosine matching.

Source: [IEEE Xplore](https://ieeexplore.ieee.org/document/10849567/)

---

### 3.2 k-NN Weighted Voting — MEDIUM PRIORITY (in ablation)

Already implemented in `ablation_sweep.py`. k=3 or k=5 with similarity-weighted voting may outperform argmax (k=1). Await ablation results.

---

### 3.3 Query TTA — MEDIUM PRIORITY (in ablation)

Embed original + hflip crop, average embeddings, L2-normalize, then match. Doubles classification time but should help with products whose reference images have different orientations.

Already in ablation sweep. Await results.

---

### 3.4 Reference TTA — DONE (precompute_refs.py)

Already implemented: 6 views per reference image (center, hflip, 4 corners). Max-sim matching (per-image, no averaging). `--angles main,front` filter to use only front-facing reference images.

---

### 3.5 AdaCos Loss — LOW PRIORITY

**Paper:** Li et al., "AdaCos: Adaptively Scaling Cosine Logits", CVPR 2019

Eliminates margin and scale hyperparameters. Adaptive scale: `s = sqrt(2) * log(C - 1)`.
For 357 classes: s ≈ 8.3 (vs our fixed scale=64).

Worth trying as ablation but requires retraining. Lower priority than inference-time tricks.

Source: [ArXiv](https://arxiv.org/abs/1905.00292)

---

### 3.6 Embedding Expansion — LOW PRIORITY

**Paper:** Ko et al., "Embedding Expansion", CVPR 2020

Linear interpolation between same-class embeddings to create synthetic training points + hard negative mining. No additional network needed.

Would need to modify training loop. Lower priority.

Source: [ArXiv](https://arxiv.org/abs/2003.02546)

---

### 3.7 DINOv2 ViT-L/14 Upgrade — BLOCKED

| Property | ViT-B/14 (current) | ViT-L/14 |
|----------|---------------------|----------|
| Parameters | 86M | 304M |
| Embedding dim | 768 | 1024 |
| FP16 size | ~172 MB | ~608 MB |
| timm name | `vit_base_patch14_dinov2.lvd142m` | `vit_large_patch14_dinov2.lvd142m` |

**Expected gain:** +2-5% Recall@1 based on DINOv2 benchmarks.

**BLOCKED:** 420 MB total weight limit. ViT-L FP16 (608 MB) + YOLO (115 MB) = 723 MB >> 420 MB.
INT8 quantization would be ~304 MB + 115 MB = 419 MB — extremely tight and may hurt fine-grained matching.

**Verdict:** Stick with ViT-B/14 unless INT8 proves lossless.

---

### 3.8 Multimodal (Image + OCR) — NOT PURSUING

Winning a Kaggle grocery competition combined image models with OCR (DistilBERT) for 94.67% F1. However, adds significant complexity and our crops may be too low-res for reliable OCR.

Source: [Springer](https://link.springer.com/article/10.1007/s00138-024-01549-9)

---

## 4. Other Datasets for Pretraining

| Dataset | Size | Task | Use for us? |
|---------|------|------|-------------|
| **SKU-110K** | 11.7K images, 1.73M boxes | Detection | **YES — detection pretrain** |
| **RP2K** | 500K+ images, 2K categories | Classification | Maybe — classification pretrain |
| **RPC** | Checkout images | Classification | Unlikely — different domain |
| **Retail-786k** | 786K images, 18K products | Matching | Maybe — entity matching pretrain |
| **Products-10K** | 150K images, 10K classes | Classification | Maybe |
| **Grocery Store Dataset** | 5.1K images, 81 classes (Swedish) | Classification | Too small |
| **SHAPE** | 50K images, 17K SKUs (Italian) | Recognition | Maybe |

**Priority:** SKU-110K for detection, RP2K or Retail-786k for classification if we need more data.

---

## 5. Prioritized Action Plan

### Tier 1 — Highest Expected Impact (do first)

| # | Action | Type | Expected Gain | Effort |
|---|--------|------|---------------|--------|
| 1 | **Run diagnostics** | Analysis | Identifies where to focus | 10 min RunPod |
| 2 | **Run ablation sweep** | Analysis | Best inference config | 5 min RunPod |
| 3 | **CAQE spatial context** | Inference trick | +1-3% cls accuracy | 2 hours coding |
| 4 | **SKU-110K pretrain** | Detection | +2-5% det mAP | Need disk space |
| 5 | **Confidence threshold sweep** | Tuning | +1-2% combined | From ablation |

### Tier 2 — Medium Impact

| # | Action | Type | Expected Gain | Effort |
|---|--------|------|---------------|--------|
| 6 | **DETR ensemble + WBF** | Detection | +1-3% det mAP | Day of training |
| 7 | **Multi-scale WBF** | Detection | +1-2% det mAP | In ablation |
| 8 | **k-NN k=3-5** | Classification | +0.5-1% cls | In ablation |
| 9 | **Query TTA** | Classification | +0.5-1% cls | In ablation |

### Tier 3 — Lower Impact / Higher Effort

| # | Action | Type | Expected Gain | Effort |
|---|--------|------|---------------|--------|
| 10 | **Copy-paste aug for rare classes** | Training | +1-2% cls on tail | Already running |
| 11 | **AdaCos loss** | Training | Unknown | Half day |
| 12 | **ViT-L upgrade** | Training | +2-5% cls | Blocked by 420 MB |
| 13 | **NMS IoU tuning** | Tuning | +0.5% det | In ablation |

### Tier BLOCKED

| # | Action | Blocker |
|---|--------|---------|
| D2 | SKU-110K pretrain | 13.6 GB download, RunPod 20 GB overlay |
| ViT-L | Weight size | 420 MB submission limit |

---

## 6. Experiment Log

> Record results here as experiments complete.

### Ablation Sweep (pending)
```
TODO: paste results table here after running on RunPod
```

### Diagnostics (pending)
```
TODO: paste key findings here
```

### Copy-Paste Augmentation (in progress)
- 185 rare classes × 30 synthetic crops
- Running on RunPod
- TODO: retrain classifier with augmented data and compare R@1

---

## 7. Key References

1. Goldman et al. (2019). "Precise Detection in Densely Packed Scenes." CVPR. [GitHub](https://github.com/eg4000/SKU110K_CVPR19)
2. Solovyev et al. (2021). "Weighted Boxes Fusion." [ArXiv](https://arxiv.org/abs/1910.13302)
3. IEEE Access (2025). "Context-Aware Fine-Grained Product Recognition on Grocery Shelves." [IEEE](https://ieeexplore.ieee.org/document/10849567/)
4. Li et al. (2019). "AdaCos: Adaptively Scaling Cosine Logits." CVPR. [ArXiv](https://arxiv.org/abs/1905.00292)
5. Ko et al. (2020). "Embedding Expansion." CVPR. [ArXiv](https://arxiv.org/abs/2003.02546)
6. Oquab et al. (2024). "DINOv2: Learning Robust Visual Features without Supervision." [ArXiv](https://arxiv.org/abs/2304.07193)
7. YOLO26 (2025). "Small-Target-Aware Label Assignment." [ArXiv](https://arxiv.org/abs/2509.25164)
8. RT-DETRv3 (WACV 2025). "Hierarchical Dense Positive Supervision." [WACV](https://openaccess.thecvf.com/content/WACV2025/papers/Wang_RT-DETRv3_Real-Time_End-to-End_Object_Detection_with_Hierarchical_Dense_Positive_Supervision_WACV_2025_paper.pdf)
9. YOLO-RACE (2025). "Enhanced YOLOv8 for SKU Detection." [Springer](https://link.springer.com/article/10.1007/s10044-025-01471-4)
10. Multimodal Grocery Recognition (2024). "Image + OCR for Product Recognition." [Springer](https://link.springer.com/article/10.1007/s00138-024-01549-9)
