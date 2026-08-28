"""Train the four-class YOLO baseline and record its configuration.

The generated dataset is expected at ``dataset/yolo``.  Training artifacts
are intentionally written below ``runs/`` (ignored by Git); the script writes
the exact command-line configuration into the run directory for the report.

Example:
    python train_yolo.py --epochs 80 --batch 16 --device 0
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "dataset" / "yolo" / "data.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics pretrained checkpoint or local path")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="CUDA device, cpu, or auto")
    parser.add_argument("--workers", type=int, default=0, help="Use 0 on Windows for reliable multiprocessing")
    parser.add_argument("--project", type=Path, default=root / "runs")
    parser.add_argument("--name", default="yolo11n_4class_baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Resolve a portable data.yaml path relative to this project even when the
    # command is launched from another working directory.
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    if not args.data.exists():
        raise SystemExit(f"Dataset YAML does not exist: {args.data}")
    if args.epochs <= 0 or args.imgsz <= 0 or args.batch == 0 or args.workers < 0:
        raise SystemExit("epochs/imgsz must be positive, batch must be non-zero, workers must be non-negative")

    args.project.mkdir(parents=True, exist_ok=True)
    started_at_utc = datetime.now(timezone.utc).isoformat()
    model = YOLO(args.model)
    results = model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        patience=args.patience,
        pretrained=True,
        amp=True,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        plots=True,
    )

    run_dir = Path(getattr(results, "save_dir", args.project / args.name))
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "started_at_utc": started_at_utc,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "model": args.model,
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "patience": args.patience,
        "resume": args.resume,
        "ultralytics": getattr(__import__("ultralytics"), "__version__", "unknown"),
    }
    (run_dir / "train_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Training complete. Run directory: {run_dir.resolve()}")
    print(f"Best checkpoint: {(run_dir / 'weights' / 'best.pt').resolve()}")


if __name__ == "__main__":
    main()
