"""Evaluate a trained detector with one-to-one class-aware IoU matching.

This script reports the course's object-level recognition accuracy in addition
to precision and recall.  The built-in ``dataset/yolo/test`` split is useful
for model diagnostics; the final acceptance number must be measured on at
least 20 independently collected objects that were never used for training.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=root / "runs" / "yolo11n_4class_v2" / "weights" / "best.pt")
    parser.add_argument("--images", type=Path, default=root / "dataset" / "yolo" / "images" / "test")
    parser.add_argument("--labels", type=Path, default=root / "dataset" / "yolo" / "labels" / "test")
    parser.add_argument("--output", type=Path, default=root / "results" / "v2_test_evaluation")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--acceptance-note",
        default="Use at least 20 independently collected objects for the final course accuracy; this split is a frame-level diagnostic.",
    )
    return parser.parse_args()


def read_ground_truth(label_path: Path, image_width: int, image_height: int) -> list[tuple[int, np.ndarray]]:
    boxes: list[tuple[int, np.ndarray]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        columns = line.split()
        if len(columns) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected five columns")
        class_id = int(columns[0])
        xc, yc, width, height = (float(value) for value in columns[1:])
        x1 = (xc - width / 2) * image_width
        y1 = (yc - height / 2) * image_height
        x2 = (xc + width / 2) * image_width
        y2 = (yc + height / 2) * image_height
        boxes.append((class_id, np.array([x1, y1, x2, y2], dtype=np.float32)))
    return boxes


def box_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection_width = max(0.0, min(float(left[2]), float(right[2])) - max(float(left[0]), float(right[0])))
    intersection_height = max(0.0, min(float(left[3]), float(right[3])) - max(float(left[1]), float(right[1])))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    right_area = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def match_detections(
    ground_truth: list[tuple[int, np.ndarray]],
    predictions: list[tuple[int, float, np.ndarray]],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    unmatched_ground_truth = set(range(len(ground_truth)))
    matches: list[tuple[int, int, float]] = []
    unmatched_predictions: list[int] = []
    for prediction_index, (predicted_class, _confidence, predicted_box) in sorted(
        enumerate(predictions), key=lambda item: item[1][1], reverse=True
    ):
        candidates = [
            (ground_truth_index, box_iou(predicted_box, ground_truth[ground_truth_index][1]))
            for ground_truth_index in unmatched_ground_truth
            if ground_truth[ground_truth_index][0] == predicted_class
        ]
        best = max(candidates, key=lambda item: item[1], default=None)
        if best is not None and best[1] >= iou_threshold:
            matches.append((prediction_index, best[0], best[1]))
            unmatched_ground_truth.remove(best[0])
        else:
            unmatched_predictions.append(prediction_index)
    return matches, sorted(unmatched_ground_truth), unmatched_predictions


def save_jpeg(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not success:
        raise OSError(f"Could not encode {path}")
    path.write_bytes(encoded.tobytes())


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise SystemExit(f"Weights do not exist: {args.weights}")
    if not args.images.is_dir() or not args.labels.is_dir():
        raise SystemExit("Image or label directory does not exist")
    if not 0 <= args.confidence <= 1 or not 0 <= args.match_iou <= 1 or not 0 <= args.nms_iou <= 1:
        raise SystemExit("Confidence and IoU thresholds must be in [0, 1]")

    args.output.mkdir(parents=True, exist_ok=True)
    error_dir = args.output / "errors"
    error_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights.resolve()))
    results = model.predict(
        source=str(args.images.resolve()),
        conf=args.confidence,
        iou=args.nms_iou,
        imgsz=args.imgsz,
        device=args.device,
        stream=True,
        save=False,
        verbose=False,
    )

    rows: list[dict[str, object]] = []
    totals = Counter()
    class_totals: dict[str, Counter] = {}
    names: dict[int, str] = {}
    for result in results:
        names = {int(key): str(value) for key, value in result.names.items()}
        image_height, image_width = result.orig_shape
        label_path = args.labels / f"{Path(result.path).stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing test label: {label_path}")
        ground_truth = read_ground_truth(label_path, image_width, image_height)
        predictions = [
            (int(class_id), float(confidence), box.astype(np.float32))
            for class_id, confidence, box in zip(
                result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.xyxy.cpu().numpy()
            )
        ]
        matches, false_negative_indices, false_positive_indices = match_detections(ground_truth, predictions, args.match_iou)

        totals.update(gt=len(ground_truth), predicted=len(predictions), correct=len(matches), false_negative=len(false_negative_indices), false_positive=len(false_positive_indices))
        for class_id, _box in ground_truth:
            class_totals.setdefault(names[class_id], Counter()).update(gt=1)
        for prediction_index, ground_truth_index, _iou in matches:
            class_id = ground_truth[ground_truth_index][0]
            class_totals.setdefault(names[class_id], Counter()).update(correct=1)
        for ground_truth_index in false_negative_indices:
            class_id = ground_truth[ground_truth_index][0]
            class_totals.setdefault(names[class_id], Counter()).update(false_negative=1)
        for prediction_index in false_positive_indices:
            class_id = predictions[prediction_index][0]
            class_totals.setdefault(names[class_id], Counter()).update(false_positive=1)

        rows.append({
            "image": Path(result.path).name,
            "ground_truth": len(ground_truth),
            "ground_truth_classes": ";".join(names[class_id] for class_id, _box in ground_truth),
            "predictions": len(predictions),
            "prediction_summary": ";".join(
                f"{names[class_id]}:{confidence:.3f}" for class_id, confidence, _box in predictions
            ),
            "correct": len(matches),
            "matched_ious": ";".join(f"{iou:.3f}" for _prediction, _ground_truth, iou in matches),
            "false_negative": len(false_negative_indices),
            "false_positive": len(false_positive_indices),
            "accuracy_percent": round(100 * len(matches) / len(ground_truth), 2) if ground_truth else "",
        })
        if false_negative_indices or false_positive_indices:
            annotated = result.plot()
            cv2.putText(annotated, "Ground truth: red", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            for class_id, box in ground_truth:
                x1, y1, x2, y2 = (int(round(float(value))) for value in box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    annotated,
                    f"GT {names[class_id]}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
            save_jpeg(error_dir / Path(result.path).name, annotated)

    with (args.output / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["image"])
        writer.writeheader()
        writer.writerows(rows)

    correct = totals["correct"]
    gt = totals["gt"]
    predicted = totals["predicted"]
    summary = {
        "weights": str(args.weights.resolve()),
        "images": len(rows),
        "ground_truth_objects": gt,
        "predicted_objects": predicted,
        "correct_objects": correct,
        "object_accuracy_percent": 100 * correct / gt if gt else 0.0,
        "precision_percent": 100 * correct / predicted if predicted else 0.0,
        "recall_percent": 100 * correct / gt if gt else 0.0,
        "false_negative": totals["false_negative"],
        "false_positive": totals["false_positive"],
        "confidence_threshold": args.confidence,
        "match_iou_threshold": args.match_iou,
        "nms_iou_threshold": args.nms_iou,
        "per_class": {name: dict(counts) for name, counts in sorted(class_totals.items())},
        "acceptance_note": args.acceptance_note,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
