# ROS2 发布快速说明

检测节点本身不直接打开摄像头。流程是：摄像头节点发布 `sensor_msgs/Image`，
`ros2_detector_node.py` 订阅图像，处理后发布 `vision_msgs/Detection2DArray`。

## 1. 启动 ROS2 环境

在 Jetson 的第一个终端：

```bash
source /opt/ros/humble/setup.bash
source ~/venvs/robotics-yolo/bin/activate
cd ~/robotics_exp1/code
```

先启动已有的 USB 摄像头或相机驱动，使图像话题出现。用下面命令确认：

```bash
ros2 topic list
ros2 topic hz /camera/image_raw
```

如果实际图像话题不是 `/camera/image_raw`，以后把命令中的话题名替换为实际名称。

## 2. 启动检测节点

仍在第一个终端运行：

```bash
python ros2_detector_node.py \
  --weights ~/robotics_exp1/weights/best.pt \
  --image-topic /camera/image_raw \
  --output-topic /detections \
  --confidence 0.35 \
  --imgsz 320 \
  --device 0
```

看到 `weights=... output_topic=/detections` 后，节点会持续发布检测结果。

## 3. 验证发布结果

打开 Jetson 的第二个终端：

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /detections --once
```

应能看到 `vision_msgs/msg/Detection2DArray`，其中包含类别名、置信度和像素坐标框。
验收时截取这条命令的输出作为 ROS2 发布证据；也可补充：

```bash
ros2 topic hz /detections
```
