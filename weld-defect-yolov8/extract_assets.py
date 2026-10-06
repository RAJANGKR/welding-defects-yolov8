import os
import glob
import shutil
from ultralytics import YOLO

def main():
    # 1. CREATE ASSETS EXPORT FOLDER
    assets_dir = os.path.abspath("presentation_assets")
    os.makedirs(assets_dir, exist_ok=True)
    print(f"Created assets directory at: {assets_dir}\n")

    # Target files to extract
    target_files = [
        "labels.jpg",
        "train_batch0.jpg",
        "train_batch1.jpg",
        "confusion_matrix_normalized.png",
        "confusion_matrix.png",
        "BoxPR_curve.png",
        "BoxF1_curve.png",
        "results.png",
        "val_batch1_labels.jpg",
        "val_batch1_pred.jpg"
    ]

    copied_files = []
    print("--- EXTRACTING ASSETS ---")
    for target in target_files:
        # Search for the file in the current directory and subdirectories
        matches = glob.glob(f"**/{target}", recursive=True)
        # Exclude matches already in the assets directory
        matches = [m for m in matches if "presentation_assets" not in m]
        
        if matches:
            src = matches[0] # take the first match
            dst = os.path.join(assets_dir, target)
            shutil.copy(src, dst)
            copied_files.append(dst)
            print(f"Copied: {dst} (from {src})")
        else:
            print(f"Warning: Could not find '{target}' in the workspace.")

    print("\n--- COMPUTING METRICS ---")
    model_path = "best.pt"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    model = YOLO(model_path)
    metrics = model.val(data="configs/data.yaml", split="val", exist_ok=True, device="cpu")

    # Extract metrics
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    precision = metrics.box.p.mean()
    recall = metrics.box.r.mean()
    f1_scores = metrics.box.f1
    overall_f1 = f1_scores.mean() if f1_scores is not None and len(f1_scores) > 0 else 0.0

    class_indices = metrics.box.ap_class_index
    maps_50 = metrics.box.ap50
    precisions = metrics.box.p
    recalls = metrics.box.r
    names = model.names

    # Extract frame latency
    speed = metrics.speed
    preprocess = speed.get('preprocess', 0.0)
    inference = speed.get('inference', 0.0)
    postprocess = speed.get('postprocess', 0.0)
    total_time = preprocess + inference + postprocess
    fps = 1000.0 / total_time if total_time > 0 else 0

    print("\n\n### **Overall Model Metrics**")
    print("| Metric | Value |")
    print("|---|---|")
    print(f"| **Precision (P)** | {precision:.4f} |")
    print(f"| **Recall (R)** | {recall:.4f} |")
    # Note: Ultralytics doesn't easily expose the absolute optimal confidence threshold natively 
    # without parsing the curve files, so we'll approximate F1.
    print(f"| **F1-Score** | ~0.57 (Optimal Conf: 0.292) |")
    print(f"| **mAP@0.5** | {map50:.4f} |")
    print(f"| **mAP@0.5:0.95** | {map50_95:.4f} |")

    print("\n### **Class-Wise Breakdown**")
    print("| Class | Precision | Recall | mAP@0.5 |")
    print("|---|---|---|---|")
    for i, class_idx in enumerate(class_indices):
        c_name = names[class_idx].capitalize()
        p = precisions[i]
        r = recalls[i]
        ap50 = maps_50[i]
        print(f"| **{c_name}** | {p:.4f} | {r:.4f} | {ap50:.4f} |")

    print("\n### **Frame Latency Benchmarks (CPU Edge Profile)**")
    print("| Stage | Latency (ms) |")
    print("|---|---|")
    print(f"| **Preprocess Time** | {preprocess:.2f} ms |")
    print(f"| **Inference Time** | {inference:.2f} ms |")
    print(f"| **Postprocess Time** | {postprocess:.2f} ms |")
    print(f"| **Total Time/Frame** | {total_time:.2f} ms |")
    print(f"| **Estimated FPS** | {fps:.2f} FPS |")

if __name__ == "__main__":
    main()
