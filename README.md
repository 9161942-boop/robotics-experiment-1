# robotics-experiment-1

## Current dataset and baseline

The project currently uses one unified four-class detector:
`mouse`, `laptop`, `cup`, and `phone`.  The source ISAT annotations are kept
under `dataset/images/<class>`.  Run the conversion after changing labels:

```powershell
python convert_isat_to_yolo.py
```

It creates `dataset/yolo/data.yaml`, YOLO labels, a `manifest.csv`, and
`conversion_summary.json`.  The 595 reviewed images are split into 421 train,
87 validation, and 87 test images using contiguous temporal blocks within each
of 19 recovered physical-object/video groups.  Every source appearance is
represented in all three splits.  This is a frame-level diagnostic split, not the final
20-independent-object acceptance test.

## Train, evaluate, and display

Ultralytics and CUDA are required for the baseline.  The first run uses a
pretrained YOLO11n checkpoint and records its configuration under `runs/`:

```powershell
python train_yolo.py --epochs 80 --batch 16 --device 0 --workers 0
python evaluate_yolo.py --weights runs/yolo11n_4class_baseline/weights/best.pt
python realtime_detect.py --weights runs/yolo11n_4class_baseline/weights/best.pt --source 0
```

`evaluate_yolo.py` performs confidence filtering and class-aware one-to-one
matching at IoU 0.50, then reports object accuracy, precision, recall, false
positives/negatives, and error images.  For the course submission, replace the
diagnostic test directory with a separately collected set containing at least
20 independent objects and preserve the resulting `summary.json`.

The first PC baseline (80 epochs, YOLO11n, CUDA RTX 4060 Laptop GPU) has
validation mAP50 0.988 and mAP50-95 0.981.  On the current 119-object frame
diagnostic test split it gives 96 correct matches, or 80.67% under the IoU
0.50 object rule.  This is a useful baseline, not the final acceptance claim:
that result predates the corrected 19-source split and is kept only as a v1
record.  The 20-object test set must be newly collected and must not share frames with
training.

The ROS2 entry point is `ros2_detector_node.py`.  On Jetson, copy the best
checkpoint to a local `weights/best.pt`, then run it with the camera topic and
inspect the result using `ros2 topic echo /detections`.
