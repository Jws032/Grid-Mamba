"""FRED RGB/event 对比工具使用的 bbox 与帧解析辅助函数。"""

from __future__ import annotations

import random
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from PIL import ImageDraw


@dataclass(frozen=True)
class BBox:
    t_s: float
    x1: float
    y1: float
    x2: float
    y2: float
    instance_id: int
    class_name: str

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


@dataclass(frozen=True)
class RgbFrame:
    member: str
    rel_t_s: float


@dataclass(frozen=True)
class SelectedBox:
    split: str
    sequence_id: str
    zip_path: Path
    reason: str
    box: BBox


def list_zips(
    root: Path,
    splits: list[str],
    sequence_ids: list[str] | None,
) -> list[tuple[str, str, Path]]:
    requested = {
        str(int(sequence_id)) if str(sequence_id).isdigit() else str(sequence_id)
        for sequence_id in sequence_ids or []
    }
    items: list[tuple[str, str, Path]] = []
    for split in splits:
        paths = sorted(
            (root / split).glob("*.zip"),
            key=lambda path: int(path.stem) if path.stem.isdigit() else path.name,
        )
        for path in paths:
            sequence_id = str(int(path.stem)) if path.stem.isdigit() else path.stem
            if requested and sequence_id not in requested:
                continue
            items.append((split, sequence_id, path))
    return items


def find_member(zf: zipfile.ZipFile, suffix: str) -> str | None:
    matches = [
        name
        for name in zf.namelist()
        if not name.endswith("/") and name.endswith(suffix)
    ]
    if not matches:
        return None
    matches.sort(key=lambda name: (name.count("/"), name))
    return matches[0]


def parse_boxes(text: str, width: int, height: int) -> list[BBox]:
    boxes: list[BBox] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or ": " not in line:
            continue
        timestamp_text, values_text = line.split(": ", 1)
        fields = values_text.split(", ")
        if len(fields) < 6:
            continue
        try:
            timestamp_s = float(timestamp_text)
            x1, y1, x2, y2 = map(float, fields[:4])
            instance_id = int(float(fields[4]))
        except ValueError:
            continue
        class_name = ", ".join(fields[5:])
        clipped_x1 = min(max(x1, 0.0), float(width - 1))
        clipped_y1 = min(max(y1, 0.0), float(height - 1))
        clipped_x2 = min(max(x2, 0.0), float(width - 1))
        clipped_y2 = min(max(y2, 0.0), float(height - 1))
        if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
            continue
        boxes.append(
            BBox(
                timestamp_s,
                clipped_x1,
                clipped_y1,
                clipped_x2,
                clipped_y2,
                instance_id,
                class_name,
            )
        )
    boxes.sort(key=lambda box: box.t_s)
    return boxes


def time_of_day_seconds(member: str) -> float | None:
    match = re.search(
        r"_(\d{2})_(\d{2})_(\d{2})\.(\d+)\.(?:jpg|jpeg|png)$",
        member,
        re.IGNORECASE,
    )
    if not match:
        return None
    hours, minutes, seconds, fraction = match.groups()
    return (
        int(hours) * 3600.0
        + int(minutes) * 60.0
        + int(seconds)
        + float("0." + fraction)
    )


def rgb_frames(zf: zipfile.ZipFile) -> list[RgbFrame]:
    members = [
        name
        for name in zf.namelist()
        if "/RGB/" in name and name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    timed = [(name, time_of_day_seconds(name)) for name in members]
    timed = [(name, timestamp) for name, timestamp in timed if timestamp is not None]
    timed.sort(key=lambda item: item[1])
    if not timed:
        return []
    start_time = timed[0][1]
    return [
        RgbFrame(name, float(timestamp - start_time))
        for name, timestamp in timed
    ]


def closest_frame(frames: list[RgbFrame], timestamp_s: float) -> tuple[RgbFrame, float]:
    lower = 0
    upper = len(frames)
    while lower < upper:
        middle = (lower + upper) // 2
        if frames[middle].rel_t_s < timestamp_s:
            lower = middle + 1
        else:
            upper = middle
    candidates: list[RgbFrame] = []
    if lower < len(frames):
        candidates.append(frames[lower])
    if lower > 0:
        candidates.append(frames[lower - 1])
    frame = min(candidates, key=lambda item: abs(item.rel_t_s - timestamp_s))
    return frame, abs(frame.rel_t_s - timestamp_s)


def select_boxes_for_sequence(
    split: str,
    sequence_id: str,
    zip_path: Path,
    boxes: list[BBox],
    count: int,
    rng: random.Random,
) -> list[SelectedBox]:
    selected: list[SelectedBox] = []
    seen: set[tuple[float, int]] = set()

    def add(reason: str, box: BBox) -> None:
        key = (box.t_s, box.instance_id)
        if key not in seen:
            selected.append(
                SelectedBox(split, sequence_id, zip_path, reason, box)
            )
            seen.add(key)

    if not boxes:
        return []
    add("largest_bbox", max(boxes, key=lambda box: box.area))
    middle_timestamp = boxes[len(boxes) // 2].t_s
    add("middle_time", min(boxes, key=lambda box: abs(box.t_s - middle_timestamp)))
    shuffled = list(boxes)
    rng.shuffle(shuffled)
    for box in shuffled:
        if len(selected) >= count:
            break
        add("random_bbox", box)
    return selected[:count]


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: BBox,
    color: str,
    width: int,
    label: str | None = None,
) -> None:
    coordinates = [box.x1, box.y1, box.x2, box.y2]
    for offset in range(width):
        draw.rectangle(
            [
                coordinates[0] - offset,
                coordinates[1] - offset,
                coordinates[2] + offset,
                coordinates[3] + offset,
            ],
            outline=color,
        )
    if label:
        draw.text((box.x1 + 3, max(0, box.y1 - 14)), label, fill=color)
