from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prediction using a trained YOLOv8 weld defect model.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--source", type=str, required=True, help="Folder or image path for prediction")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--project", type=str, default="runs/predict", help="Prediction output folder")
    parser.add_argument("--name", type=str, default="weld_predictions", help="Run name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.project).mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    print("Prediction completed.")
    print(f"Saved outputs to: {args.project}/{args.name}")


if __name__ == "__main__":
    main()
