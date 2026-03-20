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

## Output
- Best weights saved to `models/yolo_best.pt`
- Must work on L4 (24 GB VRAM) at inference
