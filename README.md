# 实验一：目标检测与识别

本项目使用自行采集和人工标注的数据，自行训练四类桌面物体检测器并部署到
Jetson Orin。整个最终训练链路未加载 COCO 或其他外部预训练权重：先以
`yolov8n.yaml` 随机初始化，在自建数据集上训练，再以自训练 checkpoint 整理四类
输出并继续训练。模型类别为 `mouse`、`laptop`、`cup` 和 `phone`。

## 项目内容

- `dataset/images/`：原始图片及 ISAT JSON 标注
- `dataset/yolo/`：595 张图片的 YOLO 标注与 train/val/test 划分（图片可由转换脚本重建）
- `weights/best.pt`：最终四类 YOLOv8n 权重
- `training/yolov8n_4class_finetune/`：训练参数、曲线和混淆矩阵
- `results/yolov8n_4class_finetune_test/`：帧级诊断结果和典型错误
- `results/yolov8n_dataset_yolo_scratch.md`：仅使用 `dataset/yolo/` 从零训练的 YOLOv8n 结果
- `results/yolov8n_coco_to_4class_finetune.md`：基于自训练 checkpoint 整理四类输出的第二阶段记录
- `results/quality_4class_best.md`：最终四类权重的验证与独立照片质量记录
- `results/acceptance_materials_checklist.md`：验收材料索引和现场演示顺序
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

训练过程从 `yolov8n.yaml` 的随机初始化开始，不加载 COCO 或其他外部 checkpoint。
第一阶段在 `dataset/yolo/data.yaml` 上训练得到内部权重；第二阶段以该内部权重为起点，
将检测头整理为四类输出后继续训练，最终验收权重为 `weights/best.pt`。因此报告中的
“从零训练”指整个训练链路没有使用外部预训练参数，而不是跳过四类输出整理步骤。

训练仅使用 train/val，不使用 test。当前划分为 train 421、val 87、test 87；
划分基于恢复出的 19 个实物/视频来源，每个来源都在三个集合中按时间块分配。

## 评估

```powershell
python evaluate_yolo.py --device 0
```

最终四类权重在 `dataset/yolo` 验证集上的 Precision、Recall、mAP@0.5 和
mAP@0.5:0.95 分别为 0.658、0.748、0.766 和 0.701；test 诊断结果保存在
`results/quality_4class_best_test_diagnostic/`。独立照片测试使用 `cup`、`laptop`、
`mouse`、`phone` 各 20 张，识别 72/80（90%），逐图结果见
`results/quality_4class_best_independent/`。独立照片指标采用图片级筛查标准，
不能替代带人工框标注的 IoU/mAP。

第一阶段随机初始化模型在 `dataset/yolo/data.yaml` 上训练 100 轮，用于验证数据转换和
训练管线。最佳模型在 87 张验证图像上达到
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
55341D55082158F799C0DA7ED813C9590EE2AE043ED2AD837DBD0B47CB0E3FE4
```

英文验收报告为 `report/experiment_report_acceptance.pdf`，并保留
`report/experiment_report_en.pdf`、`report/experiment_report_zh.pdf` 及其源文件。
Jetson 演示视频显示约 15 FPS；ROS2 节点、topic 配置和运行说明也已包含。

## 验收报告与 GitHub 证据

- 英文验收 PDF：report/experiment_report_acceptance.pdf
- LaTeX 源文件：report/experiment_report_en.tex，使用 XeLaTeX + IEEEtran 双栏论文模板直接编译
- GitHub 提交记录截图：report/github_commit_history.png
- Jetson 抽帧图：report/jetson_demo_montage.jpg
- 数据标注示例：report/dataset_annotation_examples.jpg
- 独立测试统计图：report/independent_accuracy.png
- 训练曲线与混淆矩阵：results/training/results.png、results/training/confusion_matrix_normalized.png
- 报告内含从采集、标注、转换、训练、测试到 Jetson/ROS2 发布的完整流程图与验收要求对照表
- 报告包含个人 GitHub 链接和分阶段 commit 记录，满足课程提交规范
