# YOLOv8 standard test evaluation

Command used:

```powershell
python evaluate_yolo.py --weights weights\best.pt `
  --output results\yolov8_test_runtime --confidence 0.25 `
  --match-iou 0.50 --nms-iou 0.70 --imgsz 640 --device 0
```

The custom evaluator calls Ultralytics YOLOv8 inference and performs class-aware
one-to-one matching against the test labels. On the 87-image, 104-object test
split it found 103 correct matches, one false negative, and zero false positives
(precision 100.00%, recall/object accuracy 99.04%).

For the same checkpoint, the Ultralytics standard `val(split="test", workers=0)`
run reported Precision 1.00000, Recall 0.99405, mAP@0.5 0.99315, and
mAP@0.5:0.95 0.98475. This split is a frame-level diagnostic and does not
replace the independent physical-object acceptance test.
