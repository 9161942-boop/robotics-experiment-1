# 实验一：目标检测与识别

本项目使用自行采集和标注的数据，将 COCO 预训练 `yolov8n.pt` 迁移学习为
四类桌面物体检测器，并部署到 Jetson Orin。模型类别为 `mouse`、`laptop`、
`cup` 和 `phone`。

## 项目内容

- `dataset/images/`：原始图片及 ISAT JSON 标注
- `dataset/yolo/`：595 张图片的 YOLO 标注与 train/val/test 划分（图片可由转换脚本重建）
- `weights/best.pt`：最终四类 YOLOv8n 权重
- `training/yolov8n_4class_finetune/`：训练参数、曲线和混淆矩阵
- `results/yolov8n_4class_finetune_test/`：帧级诊断结果和典型错误
- `jetson_setup/`、`docs/`：Jetson 上传脚本和运行说明
- `report/`：实验报告模板

## 安装

Windows 训练机：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Jetson 不要从 PyPI 安装普通 `torch`/`torchvision`，应使用与 JetPack 匹配的
NVIDIA 版本，具体见 [`docs/jetson_setup.md`](docs/jetson_setup.md)。

## 数据转换与训练

修改 ISAT 标注后，先重新生成 YOLO 数据集：

```powershell
python convert_isat_to_yolo.py
python train_yolo.py --epochs 100 --batch 16 --device 0 --workers 0
```

训练仅使用 train/val，不使用 test。当前划分为 train 421、val 87、test 87；
划分基于恢复出的 19 个实物/视频来源，每个来源都在三个集合中按时间块分配。

## 评估

```powershell
python evaluate_yolo.py --device 0
```

当前帧级诊断集有 87 张图像、104 个标注目标，结果为 103 个正确、1 个漏检、
0 个误检，对象级诊断准确率 99.04%。由于相邻视频帧具有相关性，这个结果只能
说明模型可用，不能代替课程要求的至少 20 个独立实物测试。独立测试请填写
`results/independent_test_template.csv`。

## Jetson 实时识别

上传文件：

```powershell
.\jetson_setup\deploy_to_jetson.ps1 -JetsonIp 192.168.43.30 -JetsonUser jetson
```

在 Jetson 图形终端运行：

```bash
cd ~/robotics_exp1/code
source ~/venvs/robotics-yolo/bin/activate
python realtime_detect.py \
  --weights ~/robotics_exp1/weights/best.pt \
  --source 0 --imgsz 320 --confidence 0.35 --device 0 \
  --camera-width 640 --camera-height 480 --camera-fps 30 \
  --output ~/robotics_exp1/results/demo.mp4
```

画面会显示检测框、类别、置信度、物体数量和滑动平均 FPS。按 `q` 退出；
SSH 无图形环境时添加 `--headless`。

## ROS2

节点订阅 `sensor_msgs/Image`，发布 `vision_msgs/Detection2DArray`：

```bash
python ros2_detector_node.py \
  --weights ~/robotics_exp1/weights/best.pt \
  --image-topic /camera/image_raw --output-topic /detections --device 0
ros2 topic echo /detections --once
```

## 最终权重

`weights/best.pt` SHA-256：

```text
595C8FD32BE2A733BF0845A5FC89847DAD425A1FDEC249C0375E0DEA9B9CFBA5
```

提交前仍需补齐独立实物测试记录、Jetson 平均 FPS、ROS2 topic 截图、结果视频
和完整实验报告。
