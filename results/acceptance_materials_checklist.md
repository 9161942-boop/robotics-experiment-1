# 实验一验收材料清单

## 已准备材料

| 验收内容 | 材料位置 | 当前状态 |
| --- | --- | --- |
| 四类桌面物体数据集与标注 | `dataset/yolo/` | 已准备，595 张图像，train/val/test 均有 YOLO 标签 |
| 80 张独立实物照片 | `dataset/independent_test/` | 已准备，mouse/laptop/cup/phone 各 20 张 |
| 四类模型 | `models/best.pt` | 已准备，`nc=4` |
| 训练参数与曲线 | `results/training/` | 已准备，100 epochs、640 输入、batch 16 |
| 独立照片测试记录 | `results/independent_test_80/` | 已准备，72/80 = 90% |
| 典型错误案例 | `results/independent_test_80/errors/`、`results/evaluation/errors/` | 已准备 |
| Jetson 演示视频 | `results/videos/jetson_demo.mp4` | 已准备，画面显示检测框、类别、置信度、物体数和约 15 FPS |
| 实时检测程序 | `realtime_detect.py` | 已准备 |
| ROS2 发布程序 | `ros2_detector_node.py` | 已准备，发布 `/detections` |
| Jetson 部署与运行说明 | `docs/jetson_setup.md`、`docs/ros2_quickstart.md` | 已准备 |
| 实验报告 | `report/experiment_report.md` | 已整理 |

## 现场演示顺序

1. 展示四类模型和类别表：`mouse`、`laptop`、`cup`、`phone`。
2. 展示数据集目录和标注文件，说明 595 张训练数据与 80 张独立测试照片分开保存。
3. 运行 `realtime_detect.py`，展示摄像头画面中的检测框、类别、置信度、物体数量和 FPS。
4. 播放 `results/videos/jetson_demo.mp4`，画面显示约 15 FPS，并展示 mouse 与 laptop 同时出现的片段。
5. 展示 `results/independent_test_80/summary.json`，说明独立照片识别率为 72/80 = 90%。
6. 展示 `results/independent_test_80/errors/` 中的错误案例。
7. 如现场需要，再按 `docs/ros2_quickstart.md` 启动 ROS2 节点并检查 `/detections`。

## 核心结论

- 类别数量：4 类，满足不少于 2 类。
- 独立照片识别率：90%，满足不低于 80%。
- Jetson 演示视频显示约 15 FPS，满足不低于 5 FPS。
- 数据集、模型、程序、视频、运行说明和报告均已放入本提交目录。

## 说明

独立照片测试采用“每张照片是否至少检测到一个期望类别”的图片级判定，
用于课程要求的实物识别率；训练验证集的 mAP 数值见 `report/experiment_report.md`，
两者不是同一个指标。
