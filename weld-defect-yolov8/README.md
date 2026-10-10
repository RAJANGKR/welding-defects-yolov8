# Automated Multi-Class Detection of Surface Weld Defects from Conventional Images Using YOLOv8

This repository contains a clean and reproducible YOLOv8 pipeline for multi-class detection of surface weld defects from conventional visible-light images. The project is designed for research documentation, academic sharing, and practical reuse.

## Overview

The model detects three surface weld defect classes:

- crack
- porosity
- spatter

The workflow covers:

- dataset validation
- `data.yaml` generation
- model training
- evaluation on validation or test split
- prediction on new images
- model export

The repository is intended to reflect the study reported in the associated manuscript, where the dataset was split into **2616 training images**, **293 validation images**, and **165 test images**, and the model was trained for **40 epochs** with **800 × 800** input size. Reported performance in the paper includes **Precision = 0.7478**, **Recall = 0.5991**, **mAP@0.5 = 0.6317**, and **mAP@0.5–0.95 = 0.3440**. These project details are consistent with the manuscript. 

## Repository Structure

```text
weld-defect-yolov8/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── configs/
│   └── data.yaml
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── validate_dataset.py
│
├── docs/
│   └── project_description.md
│
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
│
└── runs/
```

## Dataset Structure

Your dataset should be arranged exactly as follows:

```text
data/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Each image in `images/` must have a corresponding YOLO annotation text file in `labels/` with the same file stem.

Example:

```text
train/images/img_001.jpg
train/labels/img_001.txt
```

## data.yaml

The project uses a relative and GitHub-safe dataset definition:

```yaml
path: ../data
train: train/images
val: valid/images
test: test/images

nc: 3
names: ['crack', 'porosity', 'spatter']
```

This is important. Do not keep absolute local paths such as:

```yaml
train: D:\data base\improved_dataset\train\images
```

because that only works on your own system and makes the repository unusable for others.

## Installation

Create and activate an environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Run training with:

```bash
python scripts/train.py \
  --data configs/data.yaml \
  --weights yolov8n.pt \
  --epochs 40 \
  --imgsz 800 \
  --batch 4 \
  --device cpu \
  --project runs/detect \
  --name weld_defect_yolov8
```

If CUDA is available, you can use:

```bash
--device 0
```

instead of `cpu`.

## Evaluation

After training, evaluate the best model:

```bash
python scripts/evaluate.py \
  --data configs/data.yaml \
  --model runs/detect/weld_defect_yolov8/weights/best.pt \
  --split test
```

## Prediction on New Images

```bash
python scripts/predict.py \
  --model runs/detect/weld_defect_yolov8/weights/best.pt \
  --source data/test/images \
  --project runs/predict \
  --name weld_predictions
```

Predicted images will be saved automatically.

## Video Analysis

Run the Streamlit application to analyze either an image or a video:

```bash
streamlit run app.py
```

Upload an MP4, AVI, or MOV file and click **Analyze video**. The application
tracks detections across frames with ByteTrack, displays the annotated video,
plots defect counts and confidence over time, and provides CSV downloads for
the video summary and frame-level metrics. Video processing is performed
before playback; the charts are timestamped using the source video's FPS.

The video summary reports unique tracked defects rather than summing every
detection from every frame. A defect that remains visible for multiple frames
therefore contributes one tracked object to the summary.

## Dataset Validation

Before training, check whether the dataset is complete and consistent:

```bash
python scripts/validate_dataset.py --dataset-root data
```

This script checks:

- whether `images` and `labels` folders exist
- whether every image has a matching label file
- whether every label has a matching image
- whether YOLO label rows are properly formatted
- whether bounding box coordinates are normalized to `[0, 1]`

## Professional GitHub Recommendations

1. Do not upload very large raw datasets unless you have redistribution rights.
2. If the dataset is private, upload only a small demo subset or provide a dataset description.
3. Keep the code modular: train, evaluate, predict, validate.
4. Add a short project description in `docs/project_description.md`.
5. Put your article title in the README to strengthen academic visibility.
6. Include one figure later in GitHub showing sample detections.
7. Add a citation section if you publish the article.

## Suggested Citation Section

```text
If you use this repository, please cite the associated study on automated multi-class detection of surface weld defects from conventional images using deep learning.
```

## License

An MIT License template is included. You can keep it or replace it depending on your publication and dataset-sharing policy.
