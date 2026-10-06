from __future__ import annotations

import argparse
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def list_labels(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])


def parse_line(line: str):
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError("Each YOLO row must contain 5 values.")
    cls = int(parts[0])
    x, y, w, h = map(float, parts[1:])
    if min(x, y, w, h) < 0 or max(x, y, w, h) > 1:
        raise ValueError("YOLO coordinates must be normalized to the [0, 1] range.")
    return cls, x, y, w, h


def validate_split(split_dir: Path):
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    images = list_images(images_dir)
    labels = list_labels(labels_dir)

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)
    malformed = []

    for label_file in labels:
        with label_file.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    parse_line(line)
                except Exception as exc:
                    malformed.append(f"{label_file} | line {line_no} | {exc}")

    return {
        "images_count": len(images),
        "labels_count": len(labels),
        "missing_labels": missing_labels,
        "missing_images": missing_images,
        "malformed": malformed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO dataset structure and labels.")
    parser.add_argument("--dataset-root", type=str, required=True, help="Dataset root folder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)

    for split in ["train", "valid", "test"]:
        split_dir = dataset_root / split
        result = validate_split(split_dir)
        print(f"\n=== {split.upper()} ===")
        print(f"Images: {result['images_count']}")
        print(f"Labels: {result['labels_count']}")
        print(f"Missing labels: {len(result['missing_labels'])}")
        print(f"Missing images: {len(result['missing_images'])}")
        print(f"Malformed label rows: {len(result['malformed'])}")

        if result["missing_labels"]:
            print("Examples of images without labels:")
            for item in result["missing_labels"][:10]:
                print(f"  - {item}")

        if result["missing_images"]:
            print("Examples of labels without images:")
            for item in result["missing_images"][:10]:
                print(f"  - {item}")

        if result["malformed"]:
            print("Examples of malformed annotations:")
            for item in result["malformed"][:10]:
                print(f"  - {item}")


if __name__ == "__main__":
    main()
