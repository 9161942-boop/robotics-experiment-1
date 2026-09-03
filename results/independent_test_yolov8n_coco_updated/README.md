# Updated raw YOLOv8n comparison

The same updated 80-image set was evaluated with the original COCO-pretrained
`yolov8n.pt`, without project fine-tuning. The COCO `cell phone` label was used
for the expected phone class. Settings were `imgsz=640`, confidence `0.25`,
NMS IoU `0.70`, CUDA device `0`.

Result: 61/80 images recognized (76.25%): cup 14/20, laptop 19/20, mouse
17/20, phone 11/20. This is an image-level screening proxy without manually
verified boxes, so it is not IoU accuracy or final physical-object acceptance.
