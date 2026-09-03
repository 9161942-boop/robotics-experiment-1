# Evidence Analysis: Experiment 1 Object Detection and Recognition

## Analysis Question
Does the saved evidence support the course claims for four-class detection, model quality, Jetson real-time execution, result preservation, and ROS2 publication?

## Valid Findings
- The submitted detector contains four classes: mouse, laptop, cup, and phone.
- The YOLOv8n fine-tuning run completed 100 epochs. The best recorded validation mAP@0.5:0.95 was 0.98571 at epoch 76; the final epoch recorded precision 0.99417, recall 0.99405, mAP@0.5 0.99104, and mAP@0.5:0.95 0.98078.
- At confidence 0.25 and matching IoU 0.50, the frame-level test diagnostic matched 103 of 104 annotated objects, with one cup false negative and no false positives. This corresponds to 100% precision and 99.04% recall/object-level diagnostic accuracy.
- A Jetson video frame visibly reports 15.0 FPS and simultaneous detection of three objects. This is direct evidence that the implementation can exceed 5 FPS at an observed instant, but it is not a time-averaged benchmark.
- The real-time program visibly overlays class, confidence, bounding boxes, object counts, and a sliding-window FPS estimate, and it can save an MP4 result.
- The ROS2 source implements publication of `vision_msgs/Detection2DArray` on `/detections` from `/camera/image_raw`.

## Evidence Boundaries
- All train/validation/test splits contain temporal blocks from the same 19 physical-object/video source groups. Neighboring frames are correlated, so the 99.04% diagnostic must not be presented as independent-object generalization accuracy.
- Only one training seed/run is available. No inferential significance test, run-to-run standard deviation, or confidence interval is justified.
- The required independent test of at least 20 physical objects has not been entered into the saved CSV template.
- The saved video has an encoded rate of 30 FPS, but that value is not an inference throughput measurement. A representative overlay reads 15.0 FPS; a stable-window mean is still required.
- ROS2 code exists, but no saved `ros2 topic echo` output or screenshot verifies end-to-end runtime publication.

## Claim Candidates

- Claim: The trained model can detect four desktop-object classes.
  - Source evidence: checkpoint class mapping, test summary, and Jetson demo.
  - Allowed wording: "The implemented detector supports mouse, laptop, cup, and phone classes."
  - Forbidden stronger wording: "The detector recognizes arbitrary instances of all four classes with 99% real-world accuracy."
  - Uncertainty: independent-object diversity has not been measured.
  - Next check: complete the 20-object protocol.
  - Decision: keep.

- Claim: The saved frame-level diagnostic achieved 99.04% object recall/accuracy.
  - Source evidence: `results/yolov8n_4class_finetune_test/summary.json`.
  - Allowed wording: "On 87 held-out temporal frames containing 104 annotations, 103 objects were matched."
  - Forbidden stronger wording: "The final physical-object accuracy is 99.04%."
  - Uncertainty: source groups overlap across splits.
  - Next check: independent physical-object test.
  - Decision: keep with qualification.

- Claim: Jetson operation exceeded the 5 FPS threshold in the recorded demo.
  - Source evidence: `report/figures/jetson_demo_frame.jpg`, with an overlay of 15.0 FPS.
  - Allowed wording: "A representative frame in the recorded Jetson demo displayed 15.0 FPS."
  - Forbidden stronger wording: "The mean Jetson speed was 15.0 FPS."
  - Uncertainty: no stable-window average log.
  - Next check: record 30--60 seconds and save mean/minimum FPS.
  - Decision: keep with qualification.

## Decision
The model, visual overlay, video saving, and Jetson execution are sufficiently evidenced for a report draft. Final acceptance claims remain blocked on the independent 20-object table, averaged Jetson FPS, and ROS2 runtime evidence.
