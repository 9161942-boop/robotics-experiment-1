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
- `results/yolov8n_dataset_yolo_scratch.md`：仅使用 `dataset/yolo/` 从零训练的 YOLOv8n 结果
- `results/yolov8n_coco_to_4class_finetune.md`：将 COCO 模型转换为四类模型的训练结果
- `jetson_setup/`、`docs/`：Jetson 上传脚本和运行说明
- `report/`：中英文 LaTeX 实验报告、PDF 和证据分析

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

本机另运行了统一阈值为 0.25 的图片级筛查：`cup`、`laptop`、`mouse`、`phone`
各使用 `dataset/varify/` 中的 20 张图片。项目训练权重识别 47/80（58.75%），
分项为 `cup` 14/20、`laptop` 20/20、`mouse` 3/20、`phone` 10/20；原始
COCO YOLOv8n 对照模型识别 61/80（76.25%）。完整逐图结果见
`results/independent_test_runtime_updated/` 和
`results/independent_test_yolov8n_coco_updated/`，运行脚本为
[`tools/evaluate_independent_images.py`](tools/evaluate_independent_images.py)。
这只是没有人工框标注的图片级筛查代理，不能写成 IoU 准确率，因此正式验收表仍待
现场实物复核。

为验证转换后数据集能否直接训练，另使用 `dataset/yolo/data.yaml` 和未加载外部
权重的 `yolov8n.yaml` 从零训练 100 轮。最佳模型在 87 张验证图像上达到
mAP@0.5=0.991、mAP@0.5:0.95=0.966；标准 Ultralytics test 评估为
mAP@0.5=0.993、mAP@0.5:0.95=0.960。在 87 张 test 图像上，帧级诊断为
103/104 个目标匹配（精确率 98.10%，召回率 99.04%）。对 `dataset/varify/`
当前每类 20 张独立图片的图片级筛查为 41/80（51.25%），该结果没有人工框标注，
只能作为筛查代理，不能替代课程要求的独立实物验收。模型路径为
`runs/yolov8n_dataset_yolo_scratch/weights/best.pt`，完整记录见
`results/yolov8n_dataset_yolo_scratch.md`。

## Jetson 实时识别

上传文件：

```powershell
.\jetson_setup\deploy_to_jetson.ps1 -JetsonIp 192.168.43.30 -JetsonUser jetson
```

若要上传刚训练出的其他权重，可直接指定路径：

```powershell
.\jetson_setup\deploy_to_jetson.ps1 `
  -JetsonIp 192.168.43.30 -JetsonUser jetson `
  -LocalWeights ".\runs\yolov8n_4class_finetune\weights\best.pt"
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

报告已整理为 `report/experiment_report_en.pdf`（英文提交版）和
`report/experiment_report_zh.pdf`（中文对照版），对应的 LaTeX 源文件也一并保留。
课程最终验收仍需补齐独立实物测试记录、Jetson 稳定平均 FPS 和 ROS2 topic 截图；
这些缺口已在报告中明确标注，不能用帧级诊断结果替代。
