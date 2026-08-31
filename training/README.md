# Training artifacts

The reproducible training entry point is `../train_yolo.py`. The historical
YOLO11n v1/v2 baseline summaries are retained for experiment traceability.
The selected final run fine-tunes COCO-pretrained YOLOv8n for 100 epochs; its
configuration, metric CSV, curves, and confusion matrices are stored in
`yolov8n_4class_finetune/`. The selected deployment checkpoint is tracked once
at `../weights/best.pt`.

```powershell
python ../train_yolo.py --epochs 100 --batch 16 --device 0 --workers 0 --name yolov8n_4class_finetune
```

Record the SHA-256 of the checkpoint before copying it to Jetson:

```powershell
Get-FileHash ../weights/best.pt -Algorithm SHA256
```
