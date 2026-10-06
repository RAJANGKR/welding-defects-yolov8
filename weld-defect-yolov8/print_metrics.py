from ultralytics import YOLO
import os
import json

def print_metrics():
    print("="*50)
    print("   WELD DEFECT DETECTION METRICS SUMMARY")
    print("="*50)

    # 1. Load the model
    model_path = "best.pt"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please ensure training is complete.")
        return

    model = YOLO(model_path)

    # 2. Run Validation
    print("Running validation on split='val'...")
    metrics = model.val(data="configs/data.yaml", split="val", project="runs/detect", name="weld_val", exist_ok=True, device="cpu")
    
    # 3. Extract overall metrics
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    precision = metrics.box.p.mean()
    recall = metrics.box.r.mean()
    
    print("\n--- OVERALL METRICS ---")
    print(f"Precision (P):       {precision:.4f}")
    print(f"Recall (R):          {recall:.4f}")
    print(f"mAP@0.5:             {map50:.4f}")
    print(f"mAP@0.5:0.95:        {map50_95:.4f}")

    # 4. Extract class-wise metrics
    print("\n--- CLASS-WISE mAP@0.5 ---")
    class_indices = metrics.box.ap_class_index
    maps_50 = metrics.box.ap50
    names = model.names
    
    for i, class_idx in enumerate(class_indices):
        class_name = names[class_idx]
        class_map50 = maps_50[i]
        print(f"{class_name.capitalize()}: {class_map50:.4f}")

    # 5. Extract frame latency
    speed_metrics = metrics.speed
    preprocess_time = speed_metrics.get('preprocess', 0.0)
    inference_time = speed_metrics.get('inference', 0.0)
    postprocess_time = speed_metrics.get('postprocess', 0.0)
    total_time = preprocess_time + inference_time + postprocess_time
    fps = 1000.0 / total_time if total_time > 0 else 0

    print("\n--- FRAME LATENCY BENCHMARKS ---")
    print(f"Preprocess Time:     {preprocess_time:.2f} ms")
    print(f"Inference Time:      {inference_time:.2f} ms")
    print(f"Postprocess Time:    {postprocess_time:.2f} ms")
    print(f"Total Time/Frame:    {total_time:.2f} ms")
    print(f"Estimated FPS:       {fps:.2f} FPS")
    
    # 6. List file paths to generated plots
    val_dir = os.path.abspath("runs/detect/weld_val")
    print("\n--- GENERATED PLOTS (For Presentation Slides) ---")
    print(f"Normalized Confusion Matrix: {os.path.join(val_dir, 'confusion_matrix_normalized.png')}")
    print(f"Precision-Recall Curve:      {os.path.join(val_dir, 'BoxPR_curve.png')}")
    print(f"F1-Confidence Curve:         {os.path.join(val_dir, 'BoxF1_curve.png')}")
    print(f"Loss & Accuracy:             {os.path.abspath('runs/detect/weld_train/results.png')} (from training)")
    print("="*50)

if __name__ == "__main__":
    print_metrics()
