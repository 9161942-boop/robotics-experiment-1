# YOLOv8n 从零训练记录

## 训练设置

- 数据：`dataset/yolo/data.yaml`（train 421、val 87、test 87）
- 模型：`yolov8n.yaml`，随机初始化；未加载 COCO 或其他外部 checkpoint
- 轮数 / batch / 输入尺寸：100 / 16 / 640
- 设备：RTX 4060 Laptop GPU，Ultralytics 8.4.22，`workers=0`，seed=42
- 最佳权重：`runs/yolov8n_dataset_yolo_scratch/weights/best.pt`

## 结果

训练结束时验证集指标为 mAP@0.5=0.991、mAP@0.5:0.95=0.966。
标准 Ultralytics 评估（confidence=0.25、IoU=0.50）在 test 集得到
precision=0.999、recall=0.994、mAP@0.5=0.993、mAP@0.5:0.95=0.960。
固定阈值下，test 集 87 张图片、104 个标注目标的自定义帧级诊断结果为：103 个正确
匹配、1 个漏检、2 个误检，精确率 98.10%，召回率 99.04%。

`dataset/varify/` 当前每类 20 张独立图片。以“图片中至少检测到一个期望类别”为规则，
模型识别 41/80 张（51.25%），其中 cup 7/20、laptop 18/20、mouse 0/20、phone 16/20。
这些图片没有人工核验的框标注，因此该数字是图片级筛查代理，不能表述为 IoU 检测准确率。

原始日志和曲线保存在 `runs/yolov8n_dataset_yolo_scratch/`；测试集脚本输出保存在
`results/yolov8n_dataset_yolo_scratch_test/`，独立图片脚本输出保存在
`results/independent_test_runtime_dataset_yolo_scratch/`。
