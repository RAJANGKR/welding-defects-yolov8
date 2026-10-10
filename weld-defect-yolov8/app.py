import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os
import datetime
import tempfile
from pathlib import Path

from inference.video_analyzer import analyze_video

# --- Config and Setup ---
st.set_page_config(
    page_title="Weld Defect Detection AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling for Premium UI ---
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    h1, h2, h3 {
        color: #E0E6ED;
        font-family: 'Inter', sans-serif;
    }
    .metric-container {
        background-color: #1E232E;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        border-left: 4px solid #3B82F6;
    }
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        color: #3B82F6;
    }
    .metric-label {
        color: #A0AEC0;
        font-size: 1em;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .alert-banner {
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
        margin-top: 20px;
        text-align: center;
        animation: fadeIn 0.5s ease-in;
    }
    .alert-critical {
        background-color: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        border: 1px solid #EF4444;
    }
    .alert-success {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10B981;
        border: 1px solid #10B981;
    }
    .alert-warning {
        background-color: rgba(245, 158, 11, 0.2);
        color: #F59E0B;
        border: 1px solid #F59E0B;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "best.pt")
    if not os.path.exists(model_path):
        print(f"Status: Custom trained weights '{model_path}' not found, falling back to 'yolov8n.pt'.")
        model_path = "yolov8n.pt" # Fallback
    else:
        print(f"Status: Custom trained weld weights '{model_path}' loaded successfully.")
    model = YOLO(model_path)
    return model

model = load_model()

# --- Sidebar Controls ---
with st.sidebar:
    st.title("Industrial Calibration Controls")
    with st.expander("Engineering Rationale", expanded=False):
        st.write("• **Confidence Threshold:** Calibrated to 0.29 based on validated F1-optimal curve peak to balance precision and recall.")
        st.write("• **NMS IoU Threshold:** Set to 0.50 to suppress duplicate overlapping bounding boxes on single continuous flaws.")
    st.markdown("---")
    conf_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.29, step=0.01)
    iou_threshold = st.slider("IoU Threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
    st.markdown("---")
    st.info("System Status: Edge Inference Active (YOLOv8 Nano - 72 Layers)")

# --- Main App ---
st.title("🔍 Industrial Weld Defect Detection")
st.markdown("AI-Powered Quality Assurance System")

uploaded_file = st.file_uploader(
    "Upload a weld seam image or video",
    type=["jpg", "png", "jpeg", "mp4", "avi", "mov"],
)

if uploaded_file is not None:
    file_suffix = Path(uploaded_file.name).suffix.lower()
    if file_suffix in {".mp4", ".avi", ".mov"}:
        st.subheader("Video Analysis")
        st.caption("The complete video will be tracked before the annotated result and charts are shown.")
        if st.button("Analyze video", type="primary"):
            with tempfile.TemporaryDirectory(prefix="weld_video_") as work_dir:
                source_path = Path(work_dir) / uploaded_file.name
                output_path = Path(work_dir) / "annotated_video.mp4"
                source_path.write_bytes(uploaded_file.getvalue())
                progress = st.progress(0.0, text="Preparing video analysis...")

                def update_progress(value: float) -> None:
                    progress.progress(value, text=f"Analyzing video: {value:.0%}")

                try:
                    with st.spinner("Tracking defects through the video..."):
                        metrics, summary = analyze_video(
                            model,
                            source_path,
                            output_path,
                            conf_threshold,
                            iou_threshold,
                            update_progress,
                        )
                    progress.progress(1.0, text="Video analysis complete")
                    video_bytes = output_path.read_bytes()
                except (OSError, ValueError, RuntimeError) as error:
                    progress.empty()
                    st.error(f"Video analysis failed: {error}")
                    st.stop()

            unique_defects = summary["unique_defects"]
            counts = {
                "Crack": unique_defects.get("crack", 0),
                "Porosity": unique_defects.get("porosity", 0),
                "Spatter": unique_defects.get("spatter", 0),
            }
            if counts["Crack"] > 0:
                verdict = "REJECT"
            elif counts["Porosity"] > 0 or counts["Spatter"] > 0:
                verdict = "REWORK"
            else:
                verdict = "PASSED"

            st.video(video_bytes)
            st.subheader("Video Metrics")
            metric_columns = st.columns(5)
            metric_columns[0].metric("Unique defects", sum(counts.values()))
            metric_columns[1].metric("Cracks", counts["Crack"])
            metric_columns[2].metric("Porosity", counts["Porosity"])
            metric_columns[3].metric("Spatter", counts["Spatter"])
            metric_columns[4].metric("Duration", f'{summary["duration_seconds"]:.1f}s')

            if verdict == "REJECT":
                st.error("REJECT: at least one tracked crack was detected.")
            elif verdict == "REWORK":
                st.warning("REWORK: porosity or spatter was detected.")
            else:
                st.success("PASSED: no tracked weld defects were detected.")

            chart_data = metrics.set_index("timestamp_seconds")[
                ["crack", "porosity", "spatter"]
            ]
            st.subheader("Defects over time")
            st.line_chart(chart_data)

            confidence_data = metrics.set_index("timestamp_seconds")[
                ["average_confidence"]
            ]
            st.subheader("Average confidence over time")
            st.line_chart(confidence_data)

            summary_df = pd.DataFrame(
                [
                    {
                        "Filename": uploaded_file.name,
                        "Duration (seconds)": summary["duration_seconds"],
                        "FPS": summary["fps"],
                        "Frames processed": summary["frame_count"],
                        "Unique cracks": counts["Crack"],
                        "Unique porosity": counts["Porosity"],
                        "Unique spatter": counts["Spatter"],
                        "QA verdict": verdict,
                    }
                ]
            )
            st.download_button(
                "Download video summary (CSV)",
                summary_df.to_csv(index=False),
                file_name=f"QA_Video_Summary_{Path(uploaded_file.name).stem}.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download frame metrics (CSV)",
                metrics.to_csv(index=False),
                file_name=f"QA_Frame_Metrics_{Path(uploaded_file.name).stem}.csv",
                mime="text/csv",
            )
        else:
            st.info("Click **Analyze video** to process the uploaded file.")
        st.stop()

    # Load Image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Run Inference
    with st.spinner("Analyzing weld structure..."):
        results = model.predict(source=image, conf=conf_threshold, iou=iou_threshold)
        result = results[0]
        
    # Process Results
    boxes = result.boxes
    annotated_img = result.plot()[..., ::-1] # Convert BGR to RGB for Streamlit
    
    # Extract Data
    class_names = model.names
    detected_classes = [int(cls) for cls in boxes.cls.tolist()]
    confidences = boxes.conf.tolist()
    xyxy = boxes.xyxy.tolist()
    
    counts = {
        'Crack': detected_classes.count(0),
        'Porosity': detected_classes.count(1),
        'Spatter': detected_classes.count(2)
    }
    total_flaws = len(detected_classes)
    
    # --- Layout ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Raw Image")
        st.image(image, width="stretch")
        
    with col2:
        st.subheader("YOLOv8 Analysis")
        st.image(annotated_img, width="stretch")
        
    # --- Alert Banner ---
    if counts['Crack'] > 0:
        verdict = 'REJECT'
        st.markdown('<div class="alert-banner alert-critical">CRITICAL QA ALERT: Longitudinal fracture detected along weld joint. Flag part for immediate robotic rework / reject.</div>', unsafe_allow_html=True)
    elif counts['Porosity'] > 0 or counts['Spatter'] > 0:
        verdict = 'REWORK'
        st.markdown('<div class="alert-banner alert-warning">PROCESS ALERT: Surface voids/spatter detected. Check shielding gas flow and wire feed rate.</div>', unsafe_allow_html=True)
    else:
        verdict = 'PASSED'
        st.markdown('<div class="alert-banner alert-success">INSPECTION PASSED: Weld seam satisfies surface structural integrity standards.</div>', unsafe_allow_html=True)
        
    st.markdown("---")
    
    # --- Metrics ---
    st.subheader("Defect Metrics")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    def metric_card(label, value, color_hex="#3B82F6"):
        return f"""
        <div class="metric-container" style="border-left-color: {color_hex};">
            <div class="metric-value" style="color: {color_hex};">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """
        
    m_col1.markdown(metric_card("Total Flaws", total_flaws, "#F59E0B"), unsafe_allow_html=True)
    m_col2.markdown(metric_card("Cracks", counts['Crack'], "#EF4444"), unsafe_allow_html=True)
    m_col3.markdown(metric_card("Porosity", counts['Porosity'], "#8B5CF6"), unsafe_allow_html=True)
    m_col4.markdown(metric_card("Spatter", counts['Spatter'], "#10B981"), unsafe_allow_html=True)
    
    # --- Details Table ---
    table_data = []
    if total_flaws > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Detailed Detection Log")
        
        for i in range(total_flaws):
            cls_id = detected_classes[i]
            c_name = class_names[cls_id]
            if c_name.lower() == 'crack':
                risk = 'Critical'
            elif c_name.lower() == 'porosity':
                risk = 'High'
            else:
                risk = 'Moderate'
                
            box = xyxy[i]
            bbox_str = f"[{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]"
            
            table_data.append({
                "Defect Type": c_name.capitalize(),
                "Confidence": f"{confidences[i]:.2%}",
                "Hazard Risk": risk,
                "Bounding Box [X_min, Y_min, X_max, Y_max] (Pixels)": bbox_str
            })
            
        df = pd.DataFrame(table_data)
        
        # Color code risk column
        def color_risk(val):
            color = '#EF4444' if val == 'Critical' else '#F59E0B' if val == 'High' else '#10B981'
            return f'color: {color}; font-weight: bold;'
            
        st.dataframe(df.style.map(color_risk, subset=['Hazard Risk']), use_container_width=True)
        
    st.markdown("---")
    
    # Generate CSV Export
    detections_str = "; ".join([f"{d['Defect Type']} ({d['Confidence']}) at {d['Bounding Box [X_min, Y_min, X_max, Y_max] (Pixels)']}" for d in table_data]) if total_flaws > 0 else "None"
    report_data = {
        "Timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Filename": [uploaded_file.name],
        "Total Flaws": [total_flaws],
        "Cracks Count": [counts['Crack']],
        "Porosity Count": [counts['Porosity']],
        "Spatter Count": [counts['Spatter']],
        "Detections": [detections_str],
        "QA Verdict": [verdict]
    }
    
    report_df = pd.DataFrame(report_data)
    csv = report_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Export Inspection Report (CSV)",
        data=csv,
        file_name=f"QA_Report_{uploaded_file.name}.csv",
        mime="text/csv"
    )
else:
    st.info("👈 Upload an image from the sidebar or main area to begin.")
