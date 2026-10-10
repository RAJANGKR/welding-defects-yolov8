from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import cv2
import pandas as pd
from ultralytics import YOLO


def analyze_video(
    model: YOLO,
    source: str | Path,
    output_video: str | Path,
    conf: float,
    iou: float,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Track weld defects through a video and write an annotated copy."""
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if frame_count else 0.0

    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"Unable to create output video: {output_path}")

    class_names = {int(key): str(value) for key, value in model.names.items()}
    rows: list[dict[str, Any]] = []
    unique_tracks: dict[str, set[int]] = defaultdict(set)
    max_confidence: dict[str, float] = defaultdict(float)
    frame_number = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            results = model.track(
                source=frame,
                conf=conf,
                iou=iou,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )
            result = results[0]
            boxes = result.boxes
            annotated = result.plot()
            writer.write(annotated)

            class_counts = defaultdict(int)
            confidences: list[float] = []
            if boxes is not None and len(boxes):
                ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
                for class_id, confidence, track_id in zip(
                    boxes.cls.int().tolist(),
                    boxes.conf.tolist(),
                    ids,
                ):
                    defect_name = class_names.get(class_id, f"class_{class_id}")
                    normalized_name = defect_name.lower()
                    class_counts[normalized_name] += 1
                    confidences.append(float(confidence))
                    max_confidence[normalized_name] = max(
                        max_confidence[normalized_name], float(confidence)
                    )
                    if track_id is not None:
                        unique_tracks[normalized_name].add(int(track_id))

            rows.append(
                {
                    "frame": frame_number,
                    "timestamp_seconds": frame_number / fps,
                    "crack": class_counts["crack"],
                    "porosity": class_counts["porosity"],
                    "spatter": class_counts["spatter"],
                    "total_detections": sum(class_counts.values()),
                    "average_confidence": (
                        sum(confidences) / len(confidences) if confidences else 0.0
                    ),
                }
            )
            frame_number += 1
            if progress_callback and frame_count:
                progress_callback(min(frame_number / frame_count, 1.0))
    finally:
        capture.release()
        writer.release()

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise ValueError("The video contains no readable frames.")

    summary = {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_number,
        "duration_seconds": frame_number / fps,
        "unique_defects": {
            defect: len(track_ids) for defect, track_ids in unique_tracks.items()
        },
        "max_confidence": dict(max_confidence),
    }
    return metrics, summary
