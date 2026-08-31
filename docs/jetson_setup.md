# Jetson 部署与运行

## 已验证环境

- Jetson Orin，`aarch64`
- Ubuntu 20.04，Jetson Linux R35.6.4 / JetPack 5.1.6
- CUDA 11.4
- Python 3.8
- PyTorch `2.0.0a0+8aa34602.nv23.03`
- torchvision `0.15.1`，针对 Orin `sm_87` 在板端编译
- Ultralytics `8.4.22`

## 上传

Windows PowerShell：

```powershell
cd "D:\matlab\机器人学集成小组项目\实验一"
.\jetson_setup\deploy_to_jetson.ps1 -JetsonIp 192.168.43.30 -JetsonUser jetson
```

脚本上传 `weights/best.pt`、`realtime_detect.py` 和 `ros2_detector_node.py`，
并校验权重 SHA-256。密码在 SSH 提示中手动输入，不保存在脚本中。

## 实时检测

在 Jetson 图形桌面的本地终端运行：

```bash
cd ~/robotics_exp1/code
source ~/venvs/robotics-yolo/bin/activate
python realtime_detect.py \
  --weights ~/robotics_exp1/weights/best.pt \
  --source 0 --imgsz 320 --confidence 0.35 --device 0 \
  --camera-width 640 --camera-height 480 --camera-fps 30 \
  --output ~/robotics_exp1/results/demo.mp4
```

SSH 下无图形窗口时使用 `--headless`，用 `Ctrl+C` 停止：

```bash
python realtime_detect.py \
  --weights ~/robotics_exp1/weights/best.pt \
  --source 0 --imgsz 320 --device 0 --headless \
  --output ~/robotics_exp1/results/ssh_demo.mp4
```

## torchvision 说明

Jetson 的 NVIDIA PyTorch 不能配普通 PyPI torchvision wheel。如果 NMS 报
`Couldn't load custom C++ ops`，应使用 torchvision v0.15.1 源码并设置：

```bash
export CUDA_HOME=/usr/local/cuda-11.4
export FORCE_CUDA=1
unset TORCH_CUDA_ARCH_LIST
export NVCC_FLAGS="-gencode=arch=compute_87,code=sm_87"
export MAX_JOBS=2
export TORCHVISION_USE_NVJPEG=0
export TORCHVISION_USE_FFMPEG=0
export TORCHVISION_USE_VIDEO_CODEC=0
python setup.py bdist_wheel
```

生成 wheel 后必须离开 torchvision 源码目录再测试 `torchvision.ops.nms`。

## TensorRT

仅在目标 Jetson 上生成 `.engine`，不要复制其他设备生成的引擎：

```bash
cd ~/robotics_exp1
python - <<'PY'
from ultralytics import YOLO
YOLO("weights/best.pt").export(format="engine", imgsz=320, half=True, device=0)
PY
```

如果 `.pt` 未达到 5 FPS，再使用生成的 FP16 `.engine` 进行测试。
