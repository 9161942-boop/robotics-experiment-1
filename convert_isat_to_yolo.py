"""Convert ISAT polygon annotations to a reproducible YOLO detection dataset.

The source images are kept in ``dataset/images/<source-folder>`` and the
generated dataset is written to ``dataset/yolo``.  Frames are split into
contiguous temporal blocks inside every acquisition batch (for example
``cup2``).  Each class contributes to train, validation, and test; adjacent
frames are not randomly shuffled.

Usage:
    python convert_isat_to_yolo.py
    python convert_isat_to_yolo.py --images-root dataset/images \
        --output-root dataset/yolo --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CLASSES = ("mouse", "laptop", "cup", "phone")
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    json_path: Path
    source_folder: str
    source_group: str
    stem: str
    labels: tuple[tuple[int, float, float, float, float, str], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-root", type=Path, default=Path(__file__).resolve().parent / "dataset" / "images"
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path(__file__).resolve().parent / "dataset" / "yolo"
    )
    parser.add_argument(
        "--classes", default=",".join(DEFAULT_CLASSES), help="Comma-separated class order"
    )
    parser.add_argument(
        "--source-groups",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset" / "source_groups.json",
        help="JSON mapping extracted frame-number ranges back to source videos",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42, help="Recorded for reproducibility")
    return parser.parse_args()


def ensure_ratio(name: str, value: float) -> None:
    if not 0 <= value < 1:
        raise ValueError(f"{name} must be in [0, 1), got {value}")


def finite_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} is not finite: {value!r}")
    return number


def source_group_for(folder: str, stem: str, source_groups: dict[str, Any]) -> str:
    """Map an extracted frame back to its physical-object/video batch."""
    prefix = stem.rsplit("_", 1)[0] if "_" in stem else stem
    try:
        frame_index = int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        frame_index = -1
    ranges = source_groups.get(folder, {}).get(prefix, [])
    for item in ranges:
        if int(item["start"]) <= frame_index <= int(item["end"]):
            return f"{folder}:{item['group']}"
    raise ValueError(f"No source-group range for {folder}/{stem}; update dataset/source_groups.json")


def bbox_from_object(obj: dict[str, Any], width: int, height: int, context: str) -> tuple[float, float, float, float]:
    # ISAT's stored bbox is a floating-point derivative of the polygon and can
    # overshoot an image edge slightly.  The reviewed polygon is the source of
    # truth, so recompute a tight detection box directly from its vertices.
    points = obj.get("segmentation")
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError(f"{context}: missing valid segmentation")
    xy = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"{context}: malformed segmentation point")
        x = finite_number(point[0], "x")
        y = finite_number(point[1], "y")
        if x < 0 or y < 0 or x > width or y > height:
            raise ValueError(f"{context}: segmentation point outside image {width}x{height}: {point}")
        xy.append((x, y))
    raw_bbox = [min(p[0] for p in xy), min(p[1] for p in xy), max(p[0] for p in xy), max(p[1] for p in xy)]

    x1, y1, x2, y2 = (finite_number(value, "bbox") for value in raw_bbox)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{context}: bbox has non-positive area: {raw_bbox}")
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    return x1, y1, x2, y2


def read_record(
    image_path: Path,
    source_folder: str,
    class_to_id: dict[str, int],
    source_groups: dict[str, Any],
) -> ImageRecord:
    json_path = image_path.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing ISAT annotation for {image_path}")
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {json_path}") from exc

    info = data.get("info") or {}
    width = int(finite_number(info.get("width"), f"{json_path}: info.width"))
    height = int(finite_number(info.get("height"), f"{json_path}: info.height"))
    if width <= 0 or height <= 0:
        raise ValueError(f"{json_path}: invalid image size {width}x{height}")

    objects = data.get("objects")
    if not isinstance(objects, list):
        raise ValueError(f"{json_path}: objects must be a list")

    labels: list[tuple[int, float, float, float, float, str]] = []
    for object_index, obj in enumerate(objects):
        if not isinstance(obj, dict):
            raise ValueError(f"{json_path}: object {object_index} is not an object")
        category = str(obj.get("category", "")).strip()
        if category not in class_to_id:
            raise ValueError(f"{json_path}: unknown category {category!r}")
        x1, y1, x2, y2 = bbox_from_object(obj, width, height, f"{json_path} object {object_index}")
        xc = ((x1 + x2) / 2.0) / width
        yc = ((y1 + y2) / 2.0) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        labels.append((class_to_id[category], xc, yc, box_width, box_height, category))

    labels = deduplicate_labels(labels)
    return ImageRecord(
        image_path=image_path,
        json_path=json_path,
        source_folder=source_folder,
        source_group=source_group_for(source_folder, image_path.stem, source_groups),
        stem=image_path.stem,
        labels=tuple(labels),
    )


def deduplicate_labels(
    labels: list[tuple[int, float, float, float, float, str]], iou_threshold: float = 0.995
) -> list[tuple[int, float, float, float, float, str]]:
    """Remove repeated annotations of the same class for one image.

    ISAT can retain two nearly identical polygons after an edit.  YOLO treats
    them as separate targets, so collapsing only same-class boxes with a very
    high IoU prevents a duplicated label from distorting training while
    preserving legitimate overlapping objects.
    """

    def corners(label: tuple[int, float, float, float, float, str]) -> tuple[float, float, float, float]:
        _, xc, yc, width, height, _ = label
        return xc - width / 2, yc - height / 2, xc + width / 2, yc + height / 2

    def iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
        ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
        ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
        area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
        union = area_left + area_right - intersection
        return intersection / union if union else 0.0

    kept: list[tuple[int, float, float, float, float, str]] = []
    for label in labels:
        if any(label[0] == previous[0] and iou(corners(label), corners(previous)) >= iou_threshold for previous in kept):
            continue
        kept.append(label)
    return kept


def discover_records(
    images_root: Path, classes: tuple[str, ...], source_groups: dict[str, Any]
) -> list[ImageRecord]:
    class_to_id = {name: index for index, name in enumerate(classes)}
    records: list[ImageRecord] = []
    for source_folder in classes:
        folder = images_root / source_folder
        if not folder.is_dir():
            raise FileNotFoundError(f"Class image directory does not exist: {folder}")
        image_paths = sorted(
            (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda path: path.name.lower(),
        )
        if not image_paths:
            raise ValueError(f"No images found in {folder}")
        for image_path in image_paths:
            records.append(read_record(image_path, source_folder, class_to_id, source_groups))
    return records


def temporal_split(records: list[ImageRecord], val_ratio: float, test_ratio: float) -> dict[str, list[ImageRecord]]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.source_group].append(record)

    result: dict[str, list[ImageRecord]] = {split: [] for split in SPLITS}
    for group_name in sorted(grouped):
        group = sorted(grouped[group_name], key=lambda record: record.stem.lower())
        n = len(group)
        n_test = round(n * test_ratio)
        n_val = round(n * val_ratio)
        if n >= 3:
            n_test = max(1, n_test)
            n_val = max(1, n_val)
        # Keep at least one training image in every non-trivial group.
        while n - n_test - n_val < 1 and (n_test or n_val):
            if n_test >= n_val and n_test:
                n_test -= 1
            elif n_val:
                n_val -= 1
        train_end = n - n_val - n_test
        result["train"].extend(group[:train_end])
        result["val"].extend(group[train_end : train_end + n_val])
        result["test"].extend(group[train_end + n_val :])
    for split in SPLITS:
        result[split].sort(key=lambda record: (record.source_group, record.stem.lower()))
    return result


def reset_generated_dirs(output_root: Path) -> None:
    for split in SPLITS:
        for kind in ("images", "labels"):
            path = output_root / kind / split
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)


def write_dataset_yaml(output_root: Path, classes: tuple[str, ...]) -> None:
    lines = [
        "# Generated by convert_isat_to_yolo.py",
        "# Run training from the experiment project root; train_yolo.py does this automatically.",
        "path: dataset/yolo",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(classes))
    (output_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_split(record: ImageRecord, split: str, output_root: Path) -> None:
    image_destination = output_root / "images" / split / f"{record.source_folder}_{record.stem}{record.image_path.suffix.lower()}"
    label_destination = output_root / "labels" / split / f"{record.source_folder}_{record.stem}.txt"
    shutil.copy2(record.image_path, image_destination)
    label_lines = [
        f"{class_id} {xc:.6f} {yc:.6f} {box_width:.6f} {box_height:.6f}"
        for class_id, xc, yc, box_width, box_height, _ in record.labels
    ]
    label_destination.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")


def write_manifest(records_by_split: dict[str, list[ImageRecord]], output_root: Path, classes: tuple[str, ...], seed: int) -> None:
    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "source_folder", "source_group", "source_image", "output_image", "output_label", "object_count", "categories"],
        )
        writer.writeheader()
        for split in SPLITS:
            for record in records_by_split[split]:
                output_stem = f"{record.source_folder}_{record.stem}"
                categories = Counter(label[-1] for label in record.labels)
                writer.writerow({
                    "split": split,
                    "source_folder": record.source_folder,
                    "source_group": record.source_group,
                    "source_image": f"dataset/images/{record.source_folder}/{record.image_path.name}",
                    "output_image": f"images/{split}/{output_stem}{record.image_path.suffix.lower()}",
                    "output_label": f"labels/{split}/{output_stem}.txt",
                    "object_count": len(record.labels),
                    "categories": ";".join(f"{name}:{categories[name]}" for name in classes if categories[name]),
                })

    summary: dict[str, Any] = {
        "classes": list(classes),
        "seed": seed,
        "split_method": "contiguous_temporal_blocks_within_each_source_group",
        "source_group_key": "dataset/source_groups.json frame-number ranges",
        "splits": {},
    }
    for split, records in records_by_split.items():
        object_counts = Counter(label[-1] for record in records for label in record.labels)
        source_group_counts = Counter(record.source_group for record in records)
        summary["splits"][split] = {
            "images": len(records),
            "objects": sum(object_counts.values()),
            "objects_by_class": dict(object_counts),
            "source_groups": sorted({record.source_group for record in records}),
            "images_by_source_group": dict(sorted(source_group_counts.items())),
        }
    (output_root / "conversion_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    classes = tuple(name.strip() for name in args.classes.split(",") if name.strip())
    if not classes or len(set(classes)) != len(classes):
        raise SystemExit("--classes must contain unique non-empty names")
    ensure_ratio("--val-ratio", args.val_ratio)
    ensure_ratio("--test-ratio", args.test_ratio)
    if args.val_ratio + args.test_ratio >= 1:
        raise SystemExit("--val-ratio + --test-ratio must be less than 1")
    if not args.images_root.is_dir():
        raise SystemExit(f"Images root does not exist: {args.images_root}")

    if not args.source_groups.exists():
        raise SystemExit(f"Source-group config does not exist: {args.source_groups}")
    try:
        source_groups = json.loads(args.source_groups.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid source-group JSON: {args.source_groups}: {exc}") from exc
    records = discover_records(args.images_root, classes, source_groups)
    records_by_split = temporal_split(records, args.val_ratio, args.test_ratio)
    args.output_root.mkdir(parents=True, exist_ok=True)
    reset_generated_dirs(args.output_root)
    write_dataset_yaml(args.output_root, classes)
    for split, split_records in records_by_split.items():
        for record in split_records:
            write_split(record, split, args.output_root)
    write_manifest(records_by_split, args.output_root, classes, args.seed)

    total_objects = sum(len(record.labels) for record in records)
    print(f"Converted {len(records)} images and {total_objects} objects")
    print(f"Classes ({len(classes)}): {', '.join(classes)}")
    for split in SPLITS:
        split_records = records_by_split[split]
        groups = sorted({record.source_group for record in split_records})
        print(f"{split:>5}: {len(split_records):3d} images, {sum(len(r.labels) for r in split_records):3d} objects, groups={len(groups)}")
    print(f"Output: {args.output_root.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
