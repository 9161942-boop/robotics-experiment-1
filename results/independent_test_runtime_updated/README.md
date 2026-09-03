# Updated independent screening

After manual selection, the phone folder contains 20 recent smartphone images;
the old `phone_web` supplement was removed from the working set. The four
folders `cup`, `laptop`, `mouse`, and `phone` therefore contribute 20 images
each. `weights/best.pt` (the project-trained YOLOv8n checkpoint) was evaluated
with `imgsz=640`, confidence `0.25`, NMS IoU `0.70`, and CUDA device `0`.

Result: 47/80 images recognized (58.75%): cup 14/20, laptop 20/20, mouse
3/20, phone 10/20. This is an image-level screening proxy without manually
verified boxes, so it is not IoU accuracy or final physical-object acceptance.
