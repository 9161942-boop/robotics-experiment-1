# Training artifacts

The reproducible training entry point is `../train_yolo.py`.  The first PC
baseline used the settings in `baseline_config.json`; its metrics and hash are
preserved in `v1_baseline_summary.json`.  It was superseded after the source
groups were corrected.  The corrected run uses `yolo11n_4class_v2`, with its
metrics and checkpoint hash in `v2_baseline_summary.json`.
Ultralytics writes the complete curve and
checkpoint files to `../runs/yolo11n_4class_v2/`; that directory is
ignored by Git, so the large `best.pt` file is kept locally rather than
duplicated in the repository.

```powershell
python ../train_yolo.py --epochs 80 --batch 16 --device 0 --workers 0 --name yolo11n_4class_v2
```

Record the SHA-256 of the checkpoint before copying it to Jetson:

```powershell
Get-FileHash ../runs/yolo11n_4class_v2/weights/best.pt -Algorithm SHA256
```
