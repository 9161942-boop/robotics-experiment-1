"""Evaluate independent image folders with an explicit image-level criterion."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=root / "dataset" / "varify")
    parser.add_argument("--weights", type=Path, default=root / "weights" / "best.pt")
    parser.add_argument("--output", type=Path, default=root / "results" / "independent_test_runtime")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--phone-label",
        default="phone",
        help="Detector label used for the expected phone folder (COCO uses 'cell phone').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise SystemExit(f"Weights do not exist: {args.weights}")
    files = sorted(
        path
        for path in args.input_root.glob("*/*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not files:
        raise SystemExit(f"No image files found below {args.input_root}")
    if not 0 <= args.confidence <= 1 or not 0 <= args.nms_iou <= 1:
        raise SystemExit("Confidence and IoU thresholds must be in [0, 1]")

    args.output.mkdir(parents=True, exist_ok=True)
    error_dir = args.output / "errors"
    error_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights.resolve()))
    results = model.predict(
        source=[str(path) for path in files],
        conf=args.confidence,
        iou=args.nms_iou,
        imgsz=args.imgsz,
        device=args.device,
        stream=False,
        save=False,
        verbose=False,
    )

    rows: list[dict[str, object]] = []
    class_totals: dict[str, Counter] = {}
    for path, result in zip(files, results):
        folder_class = path.parent.name
        expected = "phone" if folder_class == "phone_web" else folder_class
        detections = [
            (str(result.names[int(class_id)]), float(confidence))
            for class_id, confidence in zip(result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist())
        ]
        detector_expected = args.phone_label if expected == "phone" else expected
        expected_confidences = [confidence for class_name, confidence in detections if class_name == detector_expected]
        correct = bool(expected_confidences)
        top = max(detections, key=lambda item: item[1], default=("", 0.0))
        class_totals.setdefault(expected, Counter()).update(total=1, correct=int(correct))
        row = {
            "sample_id": path.stem,
            "source": "wikimedia_commons" if folder_class == "phone_web" else "local_varify",
            "expected_class": expected,
            "image_path": str(path.relative_to(args.input_root.parent.parent)),
            "predicted_classes": ";".join(f"{name}:{confidence:.3f}" for name, confidence in detections),
            "top_class": top[0],
            "top_confidence": round(top[1], 4),
            "expected_class_detected": int(correct),
            "criterion": "expected class detected at confidence threshold",
            "detector_expected_label": detector_expected,
        }
        rows.append(row)
        if not correct:
            annotated = result.plot()
            cv2.putText(
                annotated,
                f"Expected: {row['expected_class']} | Predicted: {row['predicted_classes'] or 'none'}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )
            cv2.imwrite(str(error_dir / f"{path.stem}.jpg"), annotated)

    with (args.output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    correct = sum(int(row["expected_class_detected"]) for row in rows)
    summary = {
        "protocol": "Independent image-level recognition check: a sample is correct when at least one detection of the expected folder class survives the confidence threshold.",
        "warning": "This is a screening proxy because the images have no manually verified ground-truth boxes; it must not be reported as IoU-based object detection accuracy without annotation.",
        "weights": str(args.weights.resolve()),
        "input_root": str(args.input_root.resolve()),
        "images": total,
        "recognized": correct,
        "accuracy_percent": round(100 * correct / total, 2) if total else 0.0,
        "confidence_threshold": args.confidence,
        "nms_iou_threshold": args.nms_iou,
        "per_class": {
            name: {
                "images": counts["total"],
                "recognized": counts["correct"],
                "accuracy_percent": round(100 * counts["correct"] / counts["total"], 2),
            }
            for name, counts in sorted(class_totals.items())
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
