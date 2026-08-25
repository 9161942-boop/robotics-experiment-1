"""
从实验一/video 目录中的鼠标视频中均匀抽取图片。

默认行为：
    - 读取 video 目录下所有 mp4/avi/mov/mkv 视频；
    - 合计抽取 200 帧，而不是每个视频各抽 200 帧；
    - 图片保存到 dataset/images/mouse；
    - 如果输出目录已有同名文件，会覆盖旧图片。

运行：
    python extract_mouse_frames.py

也可以指定参数，例如抽取 300 帧并输出到自定义目录：
    python extract_mouse_frames.py --count 300 --output dataset/images/mouse
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


def extract_video_frames(video_path: Path, output_dir: Path, count: int, start_index: int) -> int:
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

        output_path = output_dir / f"mouse_{start_index:06d}.jpg"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="从鼠标视频中均匀抽取图片")
    parser.add_argument("--count", type=int, default=200, help="合计抽取的图片数量，默认 200")
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "video",
        help="视频目录，默认是当前脚本目录下的 video",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset" / "images" / "mouse",
        help="输出目录，默认是当前脚本目录下的 dataset/images/mouse",
    )
    args = parser.parse_args()

    if args.count <= 0:
        raise SystemExit("--count 必须是大于 0 的整数")
    if not args.video_dir.exists():
        raise SystemExit(f"视频目录不存在：{args.video_dir}")

    videos = find_videos(args.video_dir)
    if not videos:
        raise SystemExit(f"目录中没有找到支持的视频文件：{args.video_dir}")

    frame_counts = [get_frame_count(video) for video in videos]
    allocations = allocate_counts(frame_counts, args.count)
    total_available = sum(frame_counts)
    actual_target = min(args.count, total_available)

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"找到 {len(videos)} 个视频，总帧数约 {total_available}，计划抽取 {actual_target} 帧")
    print(f"输出目录：{args.output.resolve()}")

    next_index = 1
    for video, count in zip(videos, allocations):
        print(f"\n处理 {video.name}：抽取 {count} 帧")
        next_index = extract_video_frames(video, args.output, count, next_index)

    saved_count = next_index - 1
    print(f"\n完成：共保存 {saved_count} 张图片")
    if saved_count < args.count:
        print("提示：实际保存数量少于目标数量，可能有视频帧读取失败。")


if __name__ == "__main__":
    main()
