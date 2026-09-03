# Independent image screening

This directory records an image-level screening run of `weights/best.pt` on the
current `dataset/varify` directory.

- Input: 20 local images each for `cup`, `laptop`, and `mouse`, plus 20 public
  phone images in `dataset/varify/phone_web`.
- Runtime: Ultralytics YOLO inference, `imgsz=640`, confidence threshold
  `0.25`, NMS IoU threshold `0.70`, CUDA device `0`.
- Criterion: a sample is counted as recognized when at least one predicted box
  has the expected class at or above the confidence threshold.
- Outputs: `predictions.csv` contains one row per image; `summary.json` contains
  aggregate counts; `errors/` contains annotated failures.

The result is a screening proxy, not an IoU-based accuracy measurement: these
images do not have manually verified ground-truth boxes. The `phone_web` set is
also a public-image supplement and cannot replace the course requirement for
20 independently photographed physical phone objects. Its provenance and
license metadata are recorded in `dataset/varify/phone_web/manifest.json`.
