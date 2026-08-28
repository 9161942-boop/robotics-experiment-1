"""ROS2 image detector that publishes ``vision_msgs/Detection2DArray``.

Run this on the Jetson after installing ROS2 Humble, ``cv_bridge``,
``vision_msgs`` and Ultralytics.  The node publishes pixel-coordinate boxes in
the input image frame and keeps inference separate from the display process.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ultralytics import YOLO

try:
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
except ModuleNotFoundError as exc:
    rclpy = None
    CvBridge = Detection2D = Detection2DArray = ObjectHypothesisWithPose = None
    Image = Any
    Node = object
    ROS2_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    ROS2_IMPORT_ERROR = None


class YoloDetectionNode(Node):
    def __init__(self, weights: Path, image_topic: str, output_topic: str, confidence: float, imgsz: int, device: str) -> None:
        super().__init__("yolo_detection_node")
        self.bridge = CvBridge()
        self.model = YOLO(str(weights))
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device
        self.publisher = self.create_publisher(Detection2DArray, output_topic, 10)
        self.subscription = self.create_subscription(Image, image_topic, self.on_image, 10)
        self.get_logger().info(f"weights={weights}, image_topic={image_topic}, output_topic={output_topic}")

    def on_image(self, message: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            result = self.model.predict(frame, conf=self.confidence, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        except Exception as exc:  # Keep one bad frame from killing the ROS2 process.
            self.get_logger().error(f"inference failed: {exc}")
            return

        detections = Detection2DArray()
        detections.header = message.header
        for box, confidence, class_id in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
        ):
            x1, y1, x2, y2 = box
            detection = Detection2D()
            detection.header = message.header
            detection.bbox.center.x = (x1 + x2) / 2.0
            detection.bbox.center.y = (y1 + y2) / 2.0
            detection.bbox.size_x = max(0.0, x2 - x1)
            detection.bbox.size_y = max(0.0, y2 - y1)
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(result.names[int(class_id)])
            hypothesis.hypothesis.score = float(confidence)
            detection.results.append(hypothesis)
            detections.detections.append(detection)
        self.publisher.publish(detections)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=root / "weights" / "best.pt")
    parser.add_argument("--image-topic", default="/camera/image_raw")
    parser.add_argument("--output-topic", default="/detections")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if ROS2_IMPORT_ERROR is not None or rclpy is None:
        raise SystemExit(
            "ROS2 Python dependencies are unavailable. Source the ROS2 Humble environment "
            f"and install rclpy, cv_bridge and vision_msgs. Original error: {ROS2_IMPORT_ERROR}"
        )
    if not args.weights.exists():
        raise SystemExit(f"Weights do not exist: {args.weights}")
    if not 0 <= args.confidence <= 1:
        raise SystemExit("confidence must be in [0, 1]")
    rclpy.init()
    node = YoloDetectionNode(args.weights.resolve(), args.image_topic, args.output_topic, args.confidence, args.imgsz, args.device)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
