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
python train_yolo.py --epochs 80 --batch 16 --device 0 --workers 0 --name yolo11n_4class_v2
python evaluate_yolo.py --weights runs/yolo11n_4class_v2/weights/best.pt `
  --images dataset/yolo/images/test --labels dataset/yolo/labels/test `
  --output results/v2_test_evaluation --device 0
python realtime_detect.py --weights runs/yolo11n_4class_v2/weights/best.pt --source 0
```

`evaluate_yolo.py` performs confidence filtering and class-aware one-to-one
matching at IoU 0.50, then reports object accuracy, precision, recall, false
positives/negatives, and error images.  For the course submission, replace the
diagnostic test directory with a separately collected set containing at least
20 independent objects and preserve the resulting `summary.json`.

The superseded v1 run is retained in `training/v1_baseline_summary.json` for
process evidence.  The corrected v2 run uses the 19-source-group split and
records its configuration and metrics in `training/v2_baseline_summary.json`.
Its best validation row is epoch 63: precision 0.99337, recall 0.99405,
mAP50 0.99064, and mAP50-95 0.98244.  On the 104-object frame diagnostic
test split it gives 103 correct matches, or 99.04% under the IoU 0.50 object
rule.  This is still not the final course acceptance claim: the 20-object
test set must be newly collected and must not share frames with training.

The v2 diagnostic errors are preserved in `results/v2_typical_errors/`.  The
cup case contains two nearly identical source annotations for one visible
cup; the laptop case contains a visible mouse that is not annotated.  These
are annotation-quality issues to correct or disclose before using the images
as a final accuracy claim.

The ROS2 entry point is `ros2_detector_node.py`.  On Jetson, copy the best
checkpoint to a local `weights/best.pt`, then run it with the camera topic and
inspect the result using `ros2 topic echo /detections`.
