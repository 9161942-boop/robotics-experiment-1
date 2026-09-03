# COCO 模型转换为四类模型

以 `runs/yolov8n_dataset_yolo_scratch/weights/last.pt` 为初始化权重，将检测头从
COCO 80 类改为 `mouse`、`laptop`、`cup`、`phone` 四类，并使用
`dataset/yolo/data.yaml` 训练 100 轮。

- 最佳权重：`runs/yolov8n_coco_to_4class_finetune/weights/best.pt`
- 最后权重：`runs/yolov8n_coco_to_4class_finetune/weights/last.pt`
- 验证集：mAP@0.5=0.991，mAP@0.5:0.95=0.988
- 当前 80 张独立图片筛查（confidence=0.25）：best 55/80（68.75%），last 49/80（61.25%）

独立图片没有人工框标注，上述独立集数字是“检测到期望类别”的图片级筛查代理，
不能解释为 IoU 准确率。
