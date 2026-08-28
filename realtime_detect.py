"""Run a camera/video detector with visible FPS and object counts.

The same entry point works with a webcam index (``0``) or a video path.  It
keeps the display and inference loop small so it can later be moved to Jetson;
ROS2 publishing is deliberately kept in a separate node.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=root / "runs" / "yolo11n_4class_baseline" / "weights" / "best.pt")
    parser.add_argument("--source", default="0", help="Camera index such as 0, or a video/image path")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, default=None, help="Optional annotated video output")
    parser.add_argument("--fps-window", type=int, default=30)
    return parser.parse_args()


def source_value(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise SystemExit(f"Weights do not exist: {args.weights}")
    if not 0 <= args.confidence <= 1 or args.fps_window < 2:
        raise SystemExit("confidence must be in [0, 1] and fps-window must be at least 2")

    model = YOLO(str(args.weights.resolve()))
    capture = cv2.VideoCapture(source_value(args.source))
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    writer = None
    fps_samples: deque[float] = deque(maxlen=args.fps_window)
    previous_time = time.perf_counter()
    frames = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            result = model.predict(frame, conf=args.confidence, imgsz=args.imgsz, device=args.device, verbose=False)[0]
            annotated = result.plot()
            class_counts = Counter(int(class_id) for class_id in result.boxes.cls.cpu().tolist())
            frames += 1
            now = time.perf_counter()
            delta = now - previous_time
            previous_time = now
            if delta > 0:
                fps_samples.append(1.0 / delta)
            average_fps = sum(fps_samples) / len(fps_samples) if fps_samples else 0.0
            count_text = ", ".join(
                f"{result.names[class_id]}:{class_counts[class_id]}" for class_id in sorted(class_counts)
            ) or "none"
            cv2.rectangle(annotated, (8, 8), (410, 78), (30, 30, 30), -1)
            cv2.putText(annotated, f"FPS: {average_fps:.1f}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(annotated, f"Objects: {len(result.boxes)}  {count_text}", (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            if args.output and writer is None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                height, width = annotated.shape[:2]
                fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
                writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if writer is not None:
                writer.write(annotated)
            cv2.imshow("YOLO detection", annotated)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    print(f"Processed {frames} frames; average displayed FPS: {sum(fps_samples) / len(fps_samples):.2f}" if fps_samples else "Processed 0 frames")


if __name__ == "__main__":
    main()
