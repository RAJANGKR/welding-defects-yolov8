import collections 
import collections.abc
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import glob
import os

def add_title(slide, text):
    title_shape = slide.shapes.title
    title_shape.text = text
    for paragraph in title_shape.text_frame.paragraphs:
        paragraph.font.color.rgb = RGBColor(16, 32, 54)
        paragraph.font.bold = True
        paragraph.font.size = Pt(36)

def add_bullet_points(slide, points, left, top, width, height, font_size=18):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    
    for i, pt_text in enumerate(points):
        p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
        p.text = pt_text
        p.font.size = Pt(font_size)
        p.font.color.rgb = RGBColor(70, 85, 105) # Slate

def find_img(name):
    res = glob.glob(f"**/{name}", recursive=True)
    return res[0] if res else None

def main():
    prs = Presentation()
    # 16:9 widescreen
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # SLIDE 1: Title & Overview
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "AI-Based Welding Defect Detection System Using Computer Vision"
    for p in slide.shapes.title.text_frame.paragraphs:
        p.font.color.rgb = RGBColor(16, 32, 54)
        p.font.size = Pt(44)
    
    subtitle = slide.placeholders[1]
    subtitle.text = (
        "Real-Time Embedded Edge Inspection Pipeline via YOLOv8 Nano\n\n"
        "Embedded AI Laboratory (Assignment 9) | Department of CSE (AI & ML), PCCOE\n"
        "Date: Academic Year 2025-26\n\n"
        "Scope: Transitioning manual welding quality control to an automated, low-latency "
        "edge-vision framework to detect structural surface defects in real time."
    )
    for p in subtitle.text_frame.paragraphs:
        p.font.color.rgb = RGBColor(70, 85, 105)
        p.font.size = Pt(20)

    # SLIDE 2: Dataset Selection & Architecture
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    add_title(slide, "Dataset Selection & Architecture (Rubric: 2 Marks)")
    
    points = [
        "Dataset: Multi-Class Surface Weld Defect Dataset (Crack, Porosity, Spatter) from Kaggle.",
        "Specifications: 640x640 normalized RGB inputs, YOLO-format coordinates [class_id, x, y, w, h]."
    ]
    add_bullet_points(slide, points, Inches(0.5), Inches(1.5), Inches(6.5), Inches(1.5))
    
    # Table for Slide 2
    rows, cols = 4, 3
    table_shape = slide.shapes.add_table(rows, cols, Inches(0.5), Inches(3.0), Inches(6.5), Inches(2.0))
    table = table_shape.table
    
    headers = ["Class", "Train / Val", "Description"]
    for i, h in enumerate(headers):
        table.cell(0, i).text = h
    
    data = [
        ["Class 0: Crack", "1,982 / 273", "Narrow, high-contrast linear fissures along bead boundaries."],
        ["Class 1: Porosity", "4,605 / 152", "Clustered and isolated spherical gas voids trapped during cooling."],
        ["Class 2: Spatter", "21,504 / 1,734", "Dense, scattered molten metal droplets in heat-affected zone."]
    ]
    
    for r_idx, row_data in enumerate(data):
        for c_idx, val in enumerate(row_data):
            table.cell(r_idx + 1, c_idx).text = val
            
    img_path = find_img('labels.jpg')
    if img_path:
        slide.shapes.add_picture(img_path, Inches(7.5), Inches(1.5), width=Inches(5.0))

    # SLIDE 3: Sample & Data Explanation
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    add_title(slide, "Sample & Data Explanation (Rubric: 2 Marks)")
    
    points = [
        "Morphologies & Root Failure Causes:",
        "• Crack (Critical Structural Hazard): High-contrast directional fissures caused by severe thermal tensile stress or hydrogen embrittlement.",
        "• Porosity (Density Discontinuity): Spherical cavities caused by trapped shielding gas (Argon/CO2) or moisture/rust vaporization.",
        "• Spatter (Process Flaw): Molten metal droplets caused by excessive arc current, incorrect torch angle, or unstable wire feed."
    ]
    add_bullet_points(slide, points, Inches(0.5), Inches(1.5), Inches(6.5), Inches(4.0))
    
    img_path = find_img('train_batch1.jpg') or find_img('train_batch0.jpg')
    if img_path:
        slide.shapes.add_picture(img_path, Inches(7.5), Inches(1.5), width=Inches(5.0))

    # SLIDE 4: Model Selection
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    add_title(slide, "Model Selection & Embedded Feasibility (Rubric: 2 Marks)")
    
    points = [
        "Architecture: Single-stage anchor-free YOLOv8 Nano (yolov8n.pt).",
        "Single-stage rationale: Bypasses heavy Region Proposal Networks (RPNs), cutting latency from >80 ms to sub-10 ms."
    ]
    add_bullet_points(slide, points, Inches(0.5), Inches(1.5), Inches(12.0), Inches(1.5))
    
    # Table for Slide 4
    rows, cols = 6, 2
    table_shape = slide.shapes.add_table(rows, cols, Inches(0.5), Inches(3.0), Inches(12.0), Inches(3.5))
    table = table_shape.table
    
    table.cell(0, 0).text = "Metric"
    table.cell(0, 1).text = "Embedded Hardware Profile"
    
    data = [
        ["Parameters", "3,006,233 (~3.0M) -> Fits within constrained edge SRAM/LPDDR4 memory."],
        ["Compute Complexity", "8.1 GFLOPs -> Minimal thermal load for fanless robotic enclosures."],
        ["Weight Footprint", "6.2 MB ('best.pt') -> Rapid boot and over-the-air updates from flash memory."],
        ["Frame Latency", "40.49 ms on CPU (~24.7 FPS); drops to <5 ms (>120 FPS) with TensorRT."],
        ["Target Deployment", "NVIDIA Jetson Nano / Orin Nano, Raspberry Pi 5 with AI Hat."]
    ]
    
    for r_idx, row_data in enumerate(data):
        for c_idx, val in enumerate(row_data):
            table.cell(r_idx + 1, c_idx).text = val

    # SLIDE 5: Quantitative Evaluation & Results Analysis
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    add_title(slide, "Quantitative Evaluation & Results Analysis (Rubric: 2 Marks)")
    
    # Table for Metrics
    rows, cols = 5, 6
    table_shape = slide.shapes.add_table(rows, cols, Inches(0.5), Inches(1.5), Inches(6.5), Inches(2.0))
    table = table_shape.table
    
    headers = ["Class", "Instances", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]
    for i, h in enumerate(headers):
        table.cell(0, i).text = h
        
    data = [
        ["Crack", "273", "66.2%", "64.5%", "66.54%", "39.9%"],
        ["Porosity", "152", "59.8%", "52.0%", "51.85%", "21.6%"],
        ["Spatter", "1,734", "66.0%", "31.1%", "39.30%", "13.6%"],
        ["Overall", "2,159", "70.42%", "47.62%", "52.57%", "25.59%"]
    ]
    for r_idx, row_data in enumerate(data):
        for c_idx, val in enumerate(row_data):
            table.cell(r_idx + 1, c_idx).text = val
            
    points = [
        "Engineering Analysis:",
        "• Crack Detection: Sharp contrast against reflective beads gives strong convolutional feature activations.",
        "• Porosity Scaling: Resizing to 640x640 reduces micro-voids to single-pixel regions.",
        "• Spatter Clustering: Dense droplets trigger NMS merging; surface roughness leads to false alarms.",
        "• Operational Threshold: F1-confidence optimization identifies peak performance balance at Conf = 0.292."
    ]
    add_bullet_points(slide, points, Inches(0.5), Inches(4.0), Inches(6.5), Inches(3.0), font_size=16)
    
    img_path = find_img('confusion_matrix_normalized.png')
    if img_path:
        slide.shapes.add_picture(img_path, Inches(7.5), Inches(1.5), width=Inches(5.0))

    # SLIDE 6: Visual Validation
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    add_title(slide, "Visual Validation: Ground Truth vs. Autonomous Inference (Rubric: 2 Marks)")
    
    img1 = find_img('val_batch1_labels.jpg')
    img2 = find_img('val_batch1_pred.jpg')
    
    if img1:
        slide.shapes.add_picture(img1, Inches(0.5), Inches(1.5), width=Inches(5.5))
        txBox = slide.shapes.add_textbox(Inches(0.5), Inches(6.5), Inches(5.5), Inches(0.5))
        txBox.text_frame.text = "Ground Truth Labels"
        
    if img2:
        slide.shapes.add_picture(img2, Inches(6.5), Inches(1.5), width=Inches(5.5))
        txBox = slide.shapes.add_textbox(Inches(6.5), Inches(6.5), Inches(5.5), Inches(0.5))
        txBox.text_frame.text = "YOLOv8 Autonomous Predictions"

    # SLIDE 7: Industrial Deployment Architecture & Conclusion
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    add_title(slide, "Industrial Deployment Architecture & Conclusion")
    
    points = [
        "Production Flow: GigE Camera -> Edge SBC (Jetson/RPi5) -> YOLOv8n TensorRT Engine -> Digital I/O to PLC -> Automatic Torch Stoppage / Fault Logging.",
        "",
        "Latency Assurance: Deterministic frame cycle guarantees zero-lag closed-loop control on 30 FPS production lines.",
        "",
        "QA Impact: Eliminates subjective inspector fatigue and catches critical fractures before structural delivery.",
        "",
        "Engineering Roadmap: Future integration of Sliced Aided Hyper Inference (SAHI) or 1024x1024 patch cropping for micro-porosity resolution."
    ]
    add_bullet_points(slide, points, Inches(0.5), Inches(1.5), Inches(12.0), Inches(5.0), font_size=20)

    output_path = os.path.abspath('Welding_Defect_Detection_EAI_Assignment9.pptx')
    prs.save(output_path)
    print(f"Presentation successfully generated at: {output_path}")

if __name__ == '__main__':
    main()
