from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model for weld defect detection.")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Initial YOLO weights")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=800, help="Input image size")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda device index such as 0")
    parser.add_argument("--workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--project", type=str, default="runs/detect", help="Project output folder")
    parser.add_argument("--name", type=str, default="weld_defect_yolov8", help="Run name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.project).mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    print("Training completed.")
    print(f"Best weights should be available at: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
