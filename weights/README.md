# Deployment checkpoint

`best.pt` is the final YOLOv8n four-class checkpoint for Jetson deployment.

- Classes: `mouse`, `laptop`, `cup`, `phone`
- Input size used for training: 640
- SHA-256: `595C8FD32BE2A733BF0845A5FC89847DAD425A1FDEC249C0375E0DEA9B9CFBA5`
- Source run: `runs/yolov8n_4class_finetune/weights/best.pt`

Build TensorRT `.engine` files on the target Jetson rather than committing them,
because TensorRT engines depend on the target software and GPU environment.
