# Deployment checkpoint

`best.pt` is the selected YOLO11n four-class v2 checkpoint for Jetson deployment.

- Classes: `mouse`, `laptop`, `cup`, `phone`
- Input size used for training: 640
- SHA-256: `DFFC9873C1A365013F2F38224A4262F8E94E6657BF640576994DDE0136A1FDFE`
- Source run: `runs/yolo11n_4class_v2/weights/best.pt`

Build TensorRT `.engine` files on the target Jetson rather than committing them,
because TensorRT engines depend on the target software and GPU environment.
