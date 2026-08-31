# 实验一：目标检测与识别实验报告

## 1. 实验目的

说明四类桌面物体检测、Jetson 实时部署和 ROS2 结果发布目标。

## 2. 硬件与软件环境

- 训练机：待填写 GPU、操作系统和 CUDA
- Jetson：Orin，Ubuntu 20.04，R35.6.4 / JetPack 5.1.6
- 摄像头：待填写型号
- 模型与框架：YOLOv8n，Ultralytics 8.4.22
- ROS2：待填写发行版和消息类型

## 3. 数据集

- 类别：mouse、laptop、cup、phone
- 图像：595 张
- 划分：train 421、val 87、test 87
- 标注格式：YOLO detection
- 数据采集、场景变化和标注方法：待补充

## 4. 方法

描述视频抽帧、标注转换、COCO 预训练迁移学习、训练参数和阈值选择。

## 5. 训练与诊断结果

- 训练轮次：100
- 输入尺寸：640
- batch：16
- 帧级诊断集：104 个目标，103 个正确，准确率 99.04%
- 说明：该结果不是最终独立实物验收结果

插入 `results/training/results.png`、混淆矩阵和典型错误分析。

## 6. Jetson 实时检测

- 使用权重：`weights/best.pt`
- 摄像头输入：640x480
- 推理尺寸：待填写 320/416/640
- 平均 FPS：待实测
- 是否达到 5 FPS：待填写

## 7. 独立实物准确率

使用至少 20 个未参与训练的独立实物，按
`results/independent_test_template.csv` 记录。

- 测试物体数量：待填写
- 正确数量：待填写
- 准确率：正确数量 / 测试物体数量 x 100%
- 是否达到 80%：待填写

## 8. ROS2 发布

记录输入 topic、输出 `/detections`、消息类型
`vision_msgs/Detection2DArray`，并插入 `ros2 topic echo` 截图。

## 9. 典型错误与改进

分析漏检、误检、遮挡、反光、视角变化和小目标问题，并说明改进方向。

## 10. 结论

总结类别数量、准确率、FPS、ROS2 发布和实验要求完成情况。
