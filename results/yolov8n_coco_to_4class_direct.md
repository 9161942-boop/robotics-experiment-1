# 80 类检查点直接转换为四类输出

## 模型

- 源模型：`runs/yolov8n_coco_to_4class_finetune/weights/last.pt`
- 输出模型：`runs/yolov8n_coco_to_4class_direct/weights/last_4class_direct.pt`
- 方法：仅替换检测头的类别投影层，保留 COCO 中 `mouse`、`laptop`、`cup`、`cell phone` 四个通道；未进行微调。
- 输出类别：`mouse`、`laptop`、`cup`、`phone`

源 `last.pt` 实际保存为 `nc=80`，因此该转换是 COCO 类别通道筛选与重命名，不等同于在自定义四类数据上训练。

## 独立图片测试

测试集为 `dataset/varify`，共 80 张（每类 20 张），置信度阈值 0.25，输入尺寸 640。判定标准是图片中是否检测到至少一个期望类别目标。

| 类别 | 识别数 | 比例 |
| --- | ---: | ---: |
| cup | 17/20 | 85% |
| laptop | 19/20 | 95% |
| mouse | 18/20 | 90% |
| phone | 18/20 | 90% |
| 总计 | **72/80** | **90%** |

该结果是没有人工框标注时的图片级筛查，不应作为 IoU 或 mAP 结果使用。
