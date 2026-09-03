# `4class_best.pt` 质量检查

## 模型配置核对

- 模型：`weights/best.pt`（由 `runs/yolov8n_coco_to_4class_finetune/weights/4class_best.pt` 整理而来）
- 输出类别：`mouse`、`laptop`、`cup`、`phone`
- `nc=4`
- 参数量：3,133,436

按项目记录，该模型作为基于 `dataset/yolo` 重训得到的四类检测模型进行评估。

## `dataset/yolo` 验证集

验证集 87 张图片、104 个标注框，`imgsz=640`。

| 指标 | 数值 |
| --- | ---: |
| Precision | 0.658 |
| Recall | 0.748 |
| mAP@50 | 0.766 |
| mAP@50:95 | 0.701 |

各类别 mAP@50:95：mouse 0.723、laptop 0.877、cup 0.428、phone 0.776。

## 独立图片筛查

在 `dataset/varify` 的 80 张独立图片上，以置信度阈值 0.25 检查是否至少检测到一个期望类别：

| 类别 | 识别数 | 比例 |
| --- | ---: | ---: |
| cup | 17/20 | 85% |
| laptop | 19/20 | 95% |
| mouse | 18/20 | 90% |
| phone | 18/20 | 90% |
| 总计 | **72/80** | **90%** |

未通过图片：`04_cup.jpg`、`12_cup.jpg`、`13_cup.jpg`、`21_laptop.jpg`、`42_mouse.jpg`、`46_mouse.jpg`、`72_phone.jpg`、`80_phone.jpg`。

独立集结果是没有人工框标注时的图片级筛查，不应作为 IoU 或 mAP 结果使用。
