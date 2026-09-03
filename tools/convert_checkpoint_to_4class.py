"""Convert an 80-class YOLO checkpoint to four selected output classes.

This is a structural conversion only. It does not train the model. The
selected channels retain the source COCO class semantics and are renamed to
the project's custom class order.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from ultralytics import YOLO


COCO_IDS = (64, 63, 41, 67)  # mouse, laptop, cup, cell phone
CUSTOM_NAMES = {0: "mouse", 1: "laptop", 2: "cup", 3: "phone"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Source 80-class checkpoint")
    parser.add_argument("output", type=Path, help="Output four-class checkpoint")
    return parser.parse_args()


def replace_projection(branch: nn.Sequential) -> None:
    old = branch[-1]
    if not isinstance(old, nn.Conv2d):
        raise TypeError(f"Unexpected class projection layer: {type(old)!r}")
    new = nn.Conv2d(
        old.in_channels,
        len(COCO_IDS),
        old.kernel_size,
        old.stride,
        old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
    )
    with torch.no_grad():
        new.weight.copy_(old.weight[list(COCO_IDS)])
        if old.bias is not None:
            new.bias.copy_(old.bias[list(COCO_IDS)])
    branch[-1] = new


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise SystemExit(f"Source checkpoint does not exist: {args.source}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(str(args.source.resolve()))
    model = yolo.model
    detect = model.model[-1]
    if int(detect.nc) != 80:
        raise SystemExit(f"Expected an 80-class source model, got nc={detect.nc}")

    for branch in detect.cv3:
        replace_projection(branch)
    if hasattr(detect, "one2one_cv3"):
        for branch in detect.one2one_cv3:
            replace_projection(branch)

    detect.nc = len(COCO_IDS)
    detect.no = detect.reg_max * 4 + detect.nc
    model.nc = len(COCO_IDS)
    model.names = CUSTOM_NAMES
    if isinstance(getattr(yolo, "ckpt", None), dict):
        yolo.ckpt["nc"] = len(COCO_IDS)
        yolo.ckpt["names"] = CUSTOM_NAMES
    yolo.save(str(args.output.resolve()))

    check = YOLO(str(args.output.resolve()))
    print(f"saved={args.output.resolve()}")
    print(f"nc={check.model.nc}")
    print(f"names={check.names}")


if __name__ == "__main__":
    main()
