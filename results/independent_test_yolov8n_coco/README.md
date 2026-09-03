# Raw YOLOv8n comparison

This directory records inference with the original, unfine-tuned `yolov8n.pt`
checkpoint on the same 80-image independent screening set. The checkpoint is
the COCO-pretrained detector; it was not trained on this project's four-class
dataset. For the phone folder, the COCO label `cell phone` was counted as the
expected phone class.

- `imgsz=640`, confidence threshold `0.25`, NMS IoU threshold `0.70`, CUDA
  device `0`.
- Criterion: at least one predicted box of the expected class survives the
  confidence threshold. This is an image-level screening proxy, not IoU-based
  accuracy because the independent images have no ground-truth boxes.
- Result: 63/80 images recognized (78.75%): cup 13/20, laptop 19/20, mouse
  17/20, phone 14/20.

The result is a comparison baseline only and does not replace the course's
independently photographed physical-object acceptance test.
