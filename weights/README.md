# Deployment checkpoint

`best.pt` is the final YOLOv8n four-class checkpoint for Jetson deployment.

- Classes: `mouse`, `laptop`, `cup`, `phone`
- Input size used for training: 640
- SHA-256: `55341D55082158F799C0DA7ED813C9590EE2AE043ED2AD837DBD0B47CB0E3FE4`
- Source run: `runs/yolov8n_coco_to_4class_finetune/weights/4class_best.pt`

Build TensorRT `.engine` files on the target Jetson rather than committing them,
because TensorRT engines depend on the target software and GPU environment.
