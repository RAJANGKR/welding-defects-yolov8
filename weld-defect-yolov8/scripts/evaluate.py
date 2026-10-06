from __future__ import annotations

import argparse
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLOv8 weld defect model.")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--model", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dataset split")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    metrics = model.val(data=args.data, split=args.split)

    print("Evaluation completed.")
    print(f"Precision:   {metrics.box.mp:.4f}")
    print(f"Recall:      {metrics.box.mr:.4f}")
    print(f"mAP@0.5:     {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
