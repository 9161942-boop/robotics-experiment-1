# Figure Catalog

## Figure 1: Training Dynamics
- File: `training/yolov8n_4class_finetune/results.png`
- Purpose: show convergence of losses and detection metrics over 100 epochs.
- Key observation: losses decrease and validation metrics stabilize near the end of training.
- Caveat: one run does not show run-to-run variability.

## Figure 2: Normalized Confusion Matrix
- File: `training/yolov8n_4class_finetune/confusion_matrix_normalized.png`
- Purpose: show class-wise confusion on validation data.
- Key observation: the diagonal dominates, consistent with high validation precision and recall.
- Caveat: validation frames share physical source groups with training frames.

## Figure 3: Jetson Demonstration Frame
- File: `report/figures/jetson_demo_frame.jpg`
- Purpose: verify the live overlay and simultaneous detection behavior on Jetson.
- Key observation: mouse, laptop, and cup are shown simultaneously; boxes, confidence, counts, and 15.0 FPS are visible.
- Caveat: 15.0 FPS is a representative instantaneous/sliding-window display, not a saved benchmark mean.

## Figure 4: Typical Cup Miss
- File: `results/yolov8n_4class_finetune_test/errors/cup_cup4_000086.jpg`
- Purpose: document the sole false negative in the frame diagnostic.
- Key observation: cup is the only class with a missed annotation in this test.
- Caveat: a single miss is insufficient to estimate error probability for independent cups.
