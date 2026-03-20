# Detector Rules

## Model
- YOLO11x, class-agnostic (single class `0: product`)
- Training chain: COCO pretrained → SKU-110K pretrain → shelf fine-tune

## SKU-110K Pretraining
- Dataset: ~11,762 shelf images, ~1.7M bounding box annotations (same domain as our task)
- Annotations: CSV format with `x1,y1,x2,y2` per image → convert to YOLO `cx,cy,w,h` normalized
- User downloads manually to `data/sku110k/`, script handles conversion only
- Convert with `python -m vision_task.detection.convert_sku110k`
- Pretrain: `yolo detect train model=yolo11x.pt data=vision_task/detection/sku110k.yaml epochs=50 imgsz=1280 batch=16`

## Shelf Fine-Tune
- COCO annotations → YOLO format: collapse all 356 category_ids to class 0
- Convert with `python -m vision_task.detection.convert_coco_to_yolo`
- Fine-tune: `yolo detect train model=models/sku110k_best.pt data=vision_task/detection/shelf.yaml epochs=30 imgsz=1280 batch=8 lr0=0.001`

## Training Hyperparameters
- imgsz: 1280 (products are small on full shelf images)
- mosaic: 1.0 (critical for dense occlusion)
- mixup: 0.3
- Lower LR for shelf fine-tune (0.001 vs default 0.01)

## Evaluation
- Metric: mAP@50 and mAP@50:95
- Target: mAP@50 > 0.85
- Prioritize **recall** over precision — missing a product kills the 70% score

## DETR Comparison Experiments

### D3: HuggingFace DETR-ResNet-50 (SKU-110K pretrained)
- **Checkpoint**: `is36e/detr-resnet-50-sku110k` (HuggingFace)
- **Architecture**: DETR with ResNet-50 backbone, `num_queries=400` (important for dense shelves)
- **Pre-training**: 140 epochs decoder-only + 70 epochs full network on SKU-110K (~8K images)
- **Reported**: mAP 58.9 on SKU-110K val (single-class)
- **API**: `transformers.DetrForObjectDetection` (NOT Ultralytics)
- **Why try**: No NMS — end-to-end set prediction avoids missed detections on overlapping products
- **Plan**: Download pretrained → fine-tune decoder on shelf data → fine-tune full network
- **Fine-tune script**: `python -m vision_task.detection.train_detr` (separate from YOLO script)
- **VRAM (inference)**: DETR-R50 ~2-3 GB at native resolution — very lightweight on L4

### D4: RT-DETR via Ultralytics (optional)
- **Models**: `rtdetr-l.pt` (~32M params), `rtdetr-x.pt` (~67M params), COCO-pretrained
- **API**: Same Ultralytics API as YOLO — drop-in `--base_model rtdetr-x.pt` in `train_detector.py`
- **Why try**: Faster inference than DETR, same NMS-free benefit, but no SKU-110K pretraining available
- **VRAM (inference)**: RT-DETR-x at 1280 ~6 GB

### Decision Rule
- Run D1 (YOLO direct), D3 (HF DETR SKU-110K→shelf) in parallel
- Compare mAP@50 and **recall** — ship whichever wins
- D4 (RT-DETR) only if D3 shows DETR family is better but inference speed is a concern

## Training Scripts
- `python -m vision_task.detection.train_detector` — YOLO/RT-DETR (Ultralytics API)
- `python -m vision_task.detection.train_detr` — HuggingFace DETR (transformers API)

## Output
- Best weights saved to `models/yolo_best.pt` or `models/detr_best.pt`
- Must work on L4 (24 GB VRAM) at inference
