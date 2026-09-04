# 实验一：目标检测与识别实验报告

## 1. 实验目的

完成 mouse、laptop、cup 和 phone 四类桌面物体的目标检测，
并在 Jetson 上实现实时显示与 ROS2 结果发布。

## 2. 硬件与软件环境

- 训练机：Windows 11，NVIDIA GeForce RTX 4060 Laptop GPU，CUDA 11.8
- Jetson：Orin，Ubuntu 20.04，R35.6.4 / JetPack 5.1.6
- 摄像头：USB 摄像头，采集分辨率 640×480
- 模型与框架：YOLOv8n，Ultralytics 8.4.22
- ROS2：Humble；消息类型为 `vision_msgs/Detection2DArray`

## 3. 数据集

- 类别：mouse、laptop、cup、phone
- 图像：595 张
- 划分：train 421、val 87、test 87
- 标注格式：YOLO detection
- 数据来源：采集桌面物体视频并抽帧，使用 ISAT 标注后转换为 YOLO detection 格式；包含不同距离、角度和桌面背景。

## 4. 方法

从桌面物体视频中抽帧，使用 ISAT 完成边界框标注，再转换为 YOLO detection 格式。
训练从 `yolov8n.yaml` 的随机初始化开始，不加载 COCO 或其他外部预训练权重；第一阶段
在四类数据集上训练 100 个 epoch，随后以本实验生成的 checkpoint 整理四类检测头并继续
训练，得到最终四类模型。输入尺寸为 640，batch size 为 16，训练过程中不使用 test 集。
推理置信度阈值为 0.25（Jetson 演示使用 0.35）。

## 5. 训练与诊断结果

- 训练轮次：100
- 输入尺寸：640
- batch：16
- `dataset/yolo` 验证集 Precision：0.658，Recall：0.748
- `dataset/yolo` 验证集 mAP@50：0.766，mAP@50:95：0.701
- 第一阶段随机初始化模型 test 诊断：mAP@0.5=0.993，mAP@0.5:0.95=0.960
- 说明：第一阶段结果用于确认数据转换和训练管线；最终四类模型及独立实物验收结果见第 7 节。

训练曲线和训练期混淆矩阵保存在 `results/training/`；最终权重 test 诊断明细为
`results/evaluation_final/summary.json`、`per_image.csv` 和 `errors/`。

## 6. Jetson 实时检测

- 使用权重：`weights/best.pt`
- 摄像头输入：640x480
- 推理尺寸：以 Jetson 运行命令为准（建议 320）
- 平均 FPS：结果视频画面显示 15.0 FPS
- 是否达到 5 FPS：是（视频证据）

## 7. 独立实物准确率

使用至少 20 个未参与训练的独立实物；本实验使用 80 张独立照片（4 类×20），
详细记录见 `results/independent_test_80/`。

- 测试物体数量：80 张独立实物照片（4 类×20）
- 正确数量：72
- 准确率：72 / 80 x 100% = 90%
- 是否达到 80%：是

## 8. ROS2 发布

输入 topic：`/camera/image_raw`；输出 topic：`/detections`；消息类型：
`vision_msgs/Detection2DArray`。节点代码和运行步骤见 `docs/ros2_quickstart.md`，
现场需要时可运行 `ros2 topic echo /detections --once` 进行验证。

## 9. 典型错误与改进

独立测试未通过图片为：`04_cup.jpg`、`12_cup.jpg`、`13_cup.jpg`、
`21_laptop.jpg`、`42_mouse.jpg`、`46_mouse.jpg`、`72_phone.jpg`、`80_phone.jpg`。
主要问题是小目标、运动模糊和遮挡；可通过补充困难样本、提高输入尺寸和使用 TensorRT 优化改进。

## 10. 结论

本实验实现四类桌面物体检测，独立照片识别率为 90%，Jetson 演示视频显示 15 FPS，
超过 5 FPS 要求；同时提供 ROS2 检测节点、消息类型和运行说明。

---

本报告的正式提交版本为 `experiment_report.pdf`，由同目录的 `experiment_report.tex` 使用 XeLaTeX 直接编译。报告已补充个人 GitHub 链接、分阶段 commit 记录和 GitHub 提交页面截图，满足课程对报告模板和版本记录的要求。
