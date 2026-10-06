# Automated Multi-Class Detection of Surface Weld Defects using YOLOv8

## 📌 Overview

This project presents a deep learning-based framework for automated multi-class detection of surface weld defects using conventional images. The model is designed to detect and localize three common defect types:

* Crack
* Porosity
* Spatter

The system is implemented using YOLOv8 in a Python-based environment and is optimized for limited dataset conditions.

---

## 🎯 Objective

The main objective is to move beyond traditional binary classification and develop a multi-class object detection system capable of identifying defect locations in real-world welding scenarios.

---

## 🧠 Methodology

The workflow of the proposed system includes:

1. Data collection and annotation
2. Image preprocessing (resizing to 800×800)
3. Model training using YOLOv8
4. Validation and evaluation
5. Prediction and visualization

---

## 📊 Dataset

The dataset used in this study consists of:

* Training: 2616 images
* Validation: 293 images
* Test: 165 images

Due to GitHub storage limitations, the full dataset is not included in this repository. A small sample is provided for demonstration purposes.

📥 Full dataset: (add your Google Drive link here)

---

## ⚙️ Project Structure

```
weld-defect-yolov8/
│
├── README.md
├── requirements.txt
├── configs/
│   └── data.yaml
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
├── data/
│   ├── train/
│   ├── valid/
│   └── test/
└── sample_outputs/
```

---

## 🚀 Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training

Run the following command:

```bash
python scripts/train.py
```

---

## 📈 Evaluation

```bash
python scripts/evaluate.py
```

---

## 🔍 Prediction

```bash
python scripts/predict.py
```

---

## 📊 Results

* Precision: 0.7478
* Recall: 0.5991
* mAP@0.5: 0.6317
* mAP@0.5–0.95: 0.3440

The model shows strong performance in detecting crack defects, moderate performance for porosity, and lower accuracy for spatter due to its irregular morphology.

---

## 🖼 Sample Outputs

<img width="831" height="561" alt="image" src="https://github.com/user-attachments/assets/7fc4d784-44d9-4422-a7b8-52b84023ad21" />


---

## ⚠️ Limitations

* Dataset imbalance
* Limited resolution (800×800)
* Difficulty in detecting small defects

---

## 🔮 Future Work

* Use larger datasets
* Apply data augmentation
* Use higher resolution images
* Implement attention-based models

---

## 📜 Citation

If you use this work, please cite:

```
Benyamin Razaziyan, Elif Ağcakoca,
Automated Multi-Class Detection of Surface Weld Defects from Conventional Images Using Deep Learning
```

---

## 👨‍💻 Author

Benyamin Razaziyan
PhD Student – Civil Engineering
Sakarya University

---
## 📥 Dataset

The full dataset used in this study is not included in this repository due to size limitations.

* Train: 2616 images
* Validation: 293 images
* Test: 165 images

Download the dataset from the following link:
👉 **[Download Dataset AND MAIN FILE](https://www.kaggle.com/datasets/benyaminrazaziyan/surface-weld-defect-dataset-crak-porosityspatter)**

After downloading, extract the dataset and place it in the project directory as follows:

```
weld-defect-yolov8/
└── data/
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
## ⚙️ Setup

1. Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/weld-defect-yolov8.git
cd weld-defect-yolov8
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Download and extract the dataset into:

```
weld-defect-yolov8/data/
```

4. Run training:

```
python scripts/train.py
```
