# 自训练模型整理为四类输出的第二阶段记录

本记录的初始化权重来自本实验第一阶段的随机初始化训练结果
`runs/yolov8n_dataset_yolo_scratch/weights/last.pt`，不是 COCO 或其他外部预训练模型。
第二阶段将检测头整理为 `mouse`、`laptop`、`cup`、`phone` 四类，并继续使用
`dataset/yolo/data.yaml` 训练 100 轮，得到最终验收模型。

- 最佳权重：`runs/yolov8n_coco_to_4class_finetune/weights/best.pt`
- 最后权重：`runs/yolov8n_coco_to_4class_finetune/weights/last.pt`
- 验证集：mAP@0.5=0.991，mAP@0.5:0.95=0.988
- 当前 80 张独立图片筛查（confidence=0.25）：best 55/80（68.75%），last 49/80（61.25%）

独立图片没有人工框标注，上述独立集数字是“检测到期望类别”的图片级筛查代理，
不能解释为 IoU 准确率。
