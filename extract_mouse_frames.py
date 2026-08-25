"""
从视频中均匀抽取目标检测图片。

如果 video 目录下按类别建立了子目录（例如 video/mouse、video/cup），
脚本会按类别处理。默认每个类别抽取 200 帧，分别保存到
dataset/images/<类别名>，不会把不同类别混在同一个目录。

运行所有类别：
    python extract_mouse_frames.py

只处理指定类别：
    python extract_mouse_frames.py --classes cup,laptop,phone

如果 video 目录下直接放视频而没有类别子目录，则保留旧模式，合计抽取
200 帧并保存到 dataset/images/mouse。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


def find_videos(video_dir: Path) -> list[Path]:
    """返回视频目录中的视频文件，按文件名排序。"""
    return sorted(
        (path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda path: path.name,
    )


def find_class_dirs(video_root: Path) -> list[Path]:
    """返回包含视频文件的类别子目录。"""
    return sorted(
        (
            path
            for path in video_root.iterdir()
            if path.is_dir() and find_videos(path)
        ),
        key=lambda path: path.name.lower(),
    )


def get_frame_count(video_path: Path) -> int:
    """读取视频总帧数；读取失败时返回 0。"""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return max(frame_count, 0)


def allocate_counts(frame_counts: list[int], total_count: int) -> list[int]:
    """按视频帧数比例分配抽帧数量，并保证每个非空视频至少抽 1 帧。"""
    nonempty = [index for index, count in enumerate(frame_counts) if count > 0]
    if not nonempty:
        return [0] * len(frame_counts)

    total_available = sum(frame_counts)
    total_count = min(total_count, total_available)

    # 视频数量多于目标帧数时，优先按文件顺序给每个视频至少分配一帧。
    counts = [0] * len(frame_counts)
    minimum_count = min(len(nonempty), total_count)
    for index in nonempty[:minimum_count]:
        counts[index] = 1

    remaining = total_count - sum(counts)
    if remaining <= 0:
        return counts

    # 使用最大余数法分配剩余帧，并限制每个视频不超过自身总帧数。
    while remaining > 0:
        candidates = [index for index in nonempty if counts[index] < frame_counts[index]]
        if not candidates:
            break
        weights = [frame_counts[index] / total_available for index in candidates]
        raw_extra = [weight * remaining for weight in weights]
        extra = [int(value) for value in raw_extra]

        # 先按整数部分分配，但不超过视频容量。
        for index, amount in zip(candidates, extra):
            available = frame_counts[index] - counts[index]
            amount = min(amount, available, remaining)
            counts[index] += amount
            remaining -= amount
            if remaining == 0:
                break
        if remaining == 0:
            break

        # 整数部分全部为 0 或因容量限制仍有剩余时，逐帧按余数优先分配。
        order = sorted(
            candidates,
            key=lambda index: frame_counts[index] / total_available,
            reverse=True,
        )
        assigned = False
        for index in order:
            if counts[index] < frame_counts[index]:
                counts[index] += 1
                remaining -= 1
                assigned = True
                break
        if not assigned:
            break

    return counts


def extract_video_frames(
    video_path: Path,
    output_dir: Path,
    count: int,
    start_index: int,
    name_prefix: str,
) -> int:
    """从单个视频均匀读取 count 帧，并返回下一个图片编号。"""
    if count <= 0:
        return start_index

    frame_count = get_frame_count(video_path)
    if frame_count <= 0:
        print(f"[跳过] 无法读取视频或视频为空：{video_path.name}")
        return start_index

    # linspace 等价实现，避免额外依赖 numpy；首尾帧都会尽量覆盖。
    if count == 1:
        target_indices = [frame_count // 2]
    else:
        target_indices = [round(i * (frame_count - 1) / (count - 1)) for i in range(count)]

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"[跳过] 无法打开视频：{video_path.name}")
        return start_index

    saved_count = 0
    for target_index in target_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, target_index)
        success, frame = capture.read()
        if not success or frame is None:
            print(f"[警告] 读取失败：{video_path.name} 第 {target_index} 帧")
            continue

        output_path = output_dir / f"{name_prefix}_{start_index:06d}.jpg"
        # 先在内存中编码，再用 pathlib 写入，兼容 Windows 中文路径。
        encoded_success, encoded_image = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        )
        if encoded_success:
            output_path.write_bytes(encoded_image.tobytes())
            print(f"[{start_index:03d}] {video_path.name} -> {output_path.name}")
            start_index += 1
            saved_count += 1
        else:
            print(f"[警告] 保存失败：{output_path}")

    capture.release()
    return start_index


def get_next_index(output_dir: Path, name_prefix: str) -> tuple[int, int]:
    """返回已有图片数量和下一个可用编号，避免覆盖已有文件。"""
    existing_indices = []
    for path in output_dir.glob(f"{name_prefix}_*.jpg"):
        try:
            existing_indices.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    existing_count = len(existing_indices)
    next_index = max(existing_indices, default=0) + 1
    return existing_count, next_index


def extract_class_frames(
    class_name: str,
    class_video_dir: Path,
    output_root: Path,
    target_count: int,
) -> int:
    """从一个类别的多个视频中合计抽取 target_count 张图片。"""
    videos = find_videos(class_video_dir)
    if not videos:
        print(f"[跳过] 类别目录中没有支持的视频：{class_video_dir}")
        return 0

    output_dir = output_root / class_name
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_count, next_index = get_next_index(output_dir, class_name)
    remaining_count = max(0, target_count - existing_count)
    if remaining_count == 0:
        print(f"\n类别 {class_name} 已有 {existing_count} 张，达到目标 {target_count} 张，跳过")
        return 0

    frame_counts = [get_frame_count(video) for video in videos]
    allocations = allocate_counts(frame_counts, remaining_count)
    total_available = sum(frame_counts)
    actual_target = min(remaining_count, total_available)
    print(
        f"\n类别 {class_name}：{len(videos)} 个视频，总帧数约 {total_available}，"
        f"已有 {existing_count} 张，本次抽取 {actual_target} 张"
    )
    print(f"输出目录：{output_dir.resolve()}")

    initial_index = next_index
    for video, count in zip(videos, allocations):
        print(f"处理 {video.name}：抽取 {count} 帧")
        next_index = extract_video_frames(
            video, output_dir, count, next_index, name_prefix=class_name
        )

    saved_count = next_index - initial_index
    print(f"类别 {class_name} 完成：本次保存 {saved_count} 张，目录现有 {existing_count + saved_count} 张")
    return saved_count


def main() -> None:
    parser = argparse.ArgumentParser(description="从视频中按类别均匀抽取图片")
    parser.add_argument(
        "--count",
        type=int,
        default=200,
        help="无类别子目录时合计抽取的图片数量，默认 200",
    )
    parser.add_argument(
        "--count-per-class",
        type=int,
        default=200,
        help="有类别子目录时每个类别的目标图片数，默认 200",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="",
        help="只处理指定类别，使用逗号分隔，例如 cup,laptop,phone；默认处理全部类别",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "video",
        help="视频目录，默认是当前脚本目录下的 video",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset" / "images",
        help="类别输出根目录，默认是当前脚本目录下的 dataset/images",
    )
    args = parser.parse_args()

    if args.count <= 0 or args.count_per_class <= 0:
        raise SystemExit("--count 和 --count-per-class 都必须是大于 0 的整数")
    if not args.video_dir.exists():
        raise SystemExit(f"视频目录不存在：{args.video_dir}")

    class_dirs = find_class_dirs(args.video_dir)
    if class_dirs:
        available_classes = {path.name: path for path in class_dirs}
        requested_classes = [
            name.strip() for name in args.classes.split(",") if name.strip()
        ]
        selected_classes = requested_classes or sorted(available_classes)
        missing_classes = [name for name in selected_classes if name not in available_classes]
        if missing_classes:
            raise SystemExit(
                f"找不到类别目录：{', '.join(missing_classes)}；可用类别：{', '.join(sorted(available_classes))}"
            )

        print(f"发现类别：{', '.join(selected_classes)}")
        total_saved = 0
        for class_name in selected_classes:
            total_saved += extract_class_frames(
                class_name,
                available_classes[class_name],
                args.output,
                args.count_per_class,
            )
        print(f"\n全部完成：本次共保存 {total_saved} 张图片")
        return

    videos = find_videos(args.video_dir)
    if not videos:
        raise SystemExit(f"目录中没有找到支持的视频文件或类别子目录：{args.video_dir}")

    frame_counts = [get_frame_count(video) for video in videos]
    allocations = allocate_counts(frame_counts, args.count)
    total_available = sum(frame_counts)
    actual_target = min(args.count, total_available)

    output_dir = args.output
    if output_dir.name.lower() != "mouse":
        output_dir = output_dir / "mouse"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"找到 {len(videos)} 个视频，总帧数约 {total_available}，计划抽取 {actual_target} 帧")
    print(f"输出目录：{output_dir.resolve()}")

    next_index = 1
    for video, count in zip(videos, allocations):
        print(f"\n处理 {video.name}：抽取 {count} 帧")
        next_index = extract_video_frames(
            video, output_dir, count, next_index, name_prefix="mouse"
        )

    saved_count = next_index - 1
    print(f"\n完成：共保存 {saved_count} 张图片")
    if saved_count < args.count:
        print("提示：实际保存数量少于目标数量，可能有视频帧读取失败。")


if __name__ == "__main__":
    main()
