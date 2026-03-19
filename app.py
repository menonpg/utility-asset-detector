#!/usr/bin/env python3
"""
Gradio Web UI for Utility Asset Detector - with full observability.
"""

import json
import os

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Default class lists
DEFAULT_STRUCTURES = """utility pole
wood pole
concrete pole
transmission tower
distribution pole"""

DEFAULT_COMPONENTS = """disc insulator
ceramic insulator
suspension insulator
pin insulator
transformer
pole-mounted transformer
cross arm
power line
conductor
guy wire
fuse
lightning arrester"""

DEFAULT_CONDITIONS = """crack
rust
corrosion
burn mark
woodpecker hole
broken insulator
missing hardware
vegetation contact
lean
oil leak"""

# Global detector cache
_detector_cache = {}


def get_raw_detector():
    """Get base SAM3 predictor without our wrapper."""
    if "raw" not in _detector_cache:
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast
            
            print("Loading SAM3 model...")
            model = build_sam3_image_model(
                checkpoint_path="sam3.pt",
                device="cuda",
                eval_mode=True,
            )
            predictor = Sam3MultiClassPredictorFast(model, device="cuda")
            _detector_cache["raw"] = predictor
            print("✅ Model loaded")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return None
    return _detector_cache["raw"]


def parse_class_list(text: str) -> list[str]:
    """Parse newline-separated class list."""
    return [line.strip() for line in text.strip().split("\n") if line.strip()]


def detect_with_classes(
    image: np.ndarray,
    class_text: str,
    confidence: float,
    nms_threshold: float,
) -> tuple[np.ndarray, str, str]:
    """Run detection with custom class list."""
    if image is None:
        return None, "[]", "Upload an image"
    
    predictor = get_raw_detector()
    if predictor is None:
        return image, "[]", "❌ Model not loaded. Check DART installation."
    
    # Parse classes
    classes = parse_class_list(class_text)
    if not classes:
        return image, "[]", "❌ No classes specified"
    
    # Convert to PIL
    pil_image = Image.fromarray(image).convert("RGB")
    
    # Set classes and run detection
    predictor.set_classes(classes)
    state = predictor.set_image(pil_image)
    results = predictor.predict(state, confidence_threshold=confidence)
    
    # Extract results
    boxes = results.get('boxes', [])
    scores = results.get('scores', [])
    class_ids = results.get('class_ids', [])
    
    # Apply NMS if threshold < 1.0
    if nms_threshold < 1.0 and len(boxes) > 0:
        boxes, scores, class_ids = apply_nms(boxes, scores, class_ids, nms_threshold)
    
    # Build detections list
    detections = []
    for i, (box, score, cid) in enumerate(zip(boxes, scores, class_ids)):
        box_list = box.tolist() if hasattr(box, 'tolist') else list(box)
        detections.append({
            "id": i,
            "class": classes[cid] if cid < len(classes) else f"class_{cid}",
            "confidence": float(score),
            "bbox": [round(c, 1) for c in box_list],
        })
    
    # Sort by confidence
    detections.sort(key=lambda x: -x["confidence"])
    
    # Draw on image
    annotated = draw_boxes(pil_image.copy(), detections)
    
    # Build summary
    summary_lines = [f"**Found {len(detections)} detections**\n"]
    for det in detections:
        summary_lines.append(f"- **{det['class']}**: {det['confidence']:.0%}")
    
    return (
        np.array(annotated),
        json.dumps(detections, indent=2),
        "\n".join(summary_lines)
    )


def apply_nms(boxes, scores, class_ids, threshold):
    """Simple NMS implementation."""
    import torch
    from torchvision.ops import nms
    
    if len(boxes) == 0:
        return boxes, scores, class_ids
    
    boxes_t = torch.tensor([b.tolist() if hasattr(b, 'tolist') else b for b in boxes])
    scores_t = torch.tensor([float(s) for s in scores])
    
    keep = nms(boxes_t, scores_t, threshold)
    keep = keep.tolist()
    
    return (
        [boxes[i] for i in keep],
        [scores[i] for i in keep],
        [class_ids[i] for i in keep],
    )


def draw_boxes(image: Image.Image, detections: list) -> Image.Image:
    """Draw bounding boxes with labels."""
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", 
        "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
        "#BB8FCE", "#85C1E9", "#F8B500", "#00CED1",
    ]
    
    for i, det in enumerate(detections):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = det["bbox"]
        
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        label = f"{det['class']} {det['confidence']:.0%}"
        bbox = draw.textbbox((x1, y1 - 18), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 18), label, fill="black", font=font)
    
    return image


def run_hierarchical(
    image: np.ndarray,
    structures_text: str,
    components_text: str,
    conditions_text: str,
    struct_conf: float,
    comp_conf: float,
    cond_conf: float,
    nms_thresh: float,
) -> tuple[np.ndarray, str, str]:
    """Run full hierarchical detection."""
    if image is None:
        return None, "[]", "Upload an image"
    
    predictor = get_raw_detector()
    if predictor is None:
        return image, "[]", "❌ Model not loaded"
    
    pil_image = Image.fromarray(image).convert("RGB")
    
    structures = parse_class_list(structures_text)
    components = parse_class_list(components_text)
    conditions = parse_class_list(conditions_text)
    
    all_results = {"structures": [], "raw_detections": []}
    
    # Level 1: Detect structures
    predictor.set_classes(structures)
    state = predictor.set_image(pil_image)
    struct_results = predictor.predict(state, confidence_threshold=struct_conf)
    
    struct_boxes = struct_results.get('boxes', [])
    struct_scores = struct_results.get('scores', [])
    struct_ids = struct_results.get('class_ids', [])
    
    # Apply NMS to structures
    if nms_thresh < 1.0 and len(struct_boxes) > 0:
        struct_boxes, struct_scores, struct_ids = apply_nms(
            struct_boxes, struct_scores, struct_ids, nms_thresh
        )
    
    # For each structure, detect components
    for i, (box, score, sid) in enumerate(zip(struct_boxes, struct_scores, struct_ids)):
        box_list = box.tolist() if hasattr(box, 'tolist') else list(box)
        struct_name = structures[sid] if sid < len(structures) else f"structure_{sid}"
        
        struct_data = {
            "id": i,
            "type": struct_name,
            "confidence": float(score),
            "bbox": box_list,
            "components": [],
            "conditions": [],
        }
        
        # Crop to structure ROI with padding
        x1, y1, x2, y2 = box_list
        w, h = x2 - x1, y2 - y1
        pad = 0.1
        img_w, img_h = pil_image.size
        
        roi = [
            max(0, x1 - w * pad),
            max(0, y1 - h * pad),
            min(img_w, x2 + w * pad),
            min(img_h, y2 + h * pad),
        ]
        roi_image = pil_image.crop([int(c) for c in roi])
        
        # Level 2: Detect components in ROI
        if components:
            predictor.set_classes(components)
            comp_state = predictor.set_image(roi_image)
            comp_results = predictor.predict(comp_state, confidence_threshold=comp_conf)
            
            for cbox, cscore, cid in zip(
                comp_results.get('boxes', []),
                comp_results.get('scores', []),
                comp_results.get('class_ids', [])
            ):
                cbox_list = cbox.tolist() if hasattr(cbox, 'tolist') else list(cbox)
                # Convert ROI coords to image coords
                cbox_list = [
                    cbox_list[0] + roi[0],
                    cbox_list[1] + roi[1],
                    cbox_list[2] + roi[0],
                    cbox_list[3] + roi[1],
                ]
                struct_data["components"].append({
                    "type": components[cid] if cid < len(components) else f"comp_{cid}",
                    "confidence": float(cscore),
                    "bbox": cbox_list,
                })
        
        # Level 3: Detect conditions in ROI
        if conditions:
            predictor.set_classes(conditions)
            cond_state = predictor.set_image(roi_image)
            cond_results = predictor.predict(cond_state, confidence_threshold=cond_conf)
            
            for dbox, dscore, did in zip(
                cond_results.get('boxes', []),
                cond_results.get('scores', []),
                cond_results.get('class_ids', [])
            ):
                struct_data["conditions"].append({
                    "type": conditions[did] if did < len(conditions) else f"cond_{did}",
                    "confidence": float(dscore),
                })
        
        all_results["structures"].append(struct_data)
    
    # Draw annotations
    annotated = draw_hierarchical(pil_image.copy(), all_results)
    
    # Build summary
    summary_lines = [f"**Found {len(all_results['structures'])} structures**\n"]
    for struct in all_results["structures"]:
        summary_lines.append(f"\n### {struct['type']} ({struct['confidence']:.0%})")
        
        if struct["components"]:
            summary_lines.append("**Components:**")
            for comp in struct["components"]:
                summary_lines.append(f"- {comp['type']}: {comp['confidence']:.0%}")
        
        if struct["conditions"]:
            summary_lines.append("**Conditions:**")
            for cond in struct["conditions"]:
                summary_lines.append(f"- ⚠️ {cond['type']}: {cond['confidence']:.0%}")
    
    return (
        np.array(annotated),
        json.dumps(all_results, indent=2),
        "\n".join(summary_lines),
    )


def draw_hierarchical(image: Image.Image, results: dict) -> Image.Image:
    """Draw hierarchical detection results."""
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    struct_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
    comp_color = "#FFEAA7"
    
    for i, struct in enumerate(results.get("structures", [])):
        color = struct_colors[i % len(struct_colors)]
        x1, y1, x2, y2 = struct["bbox"]
        
        # Structure box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{struct['type']} {struct['confidence']:.0%}"
        bbox = draw.textbbox((x1, y1 - 18), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 18), label, fill="black", font=font)
        
        # Component boxes
        for comp in struct.get("components", []):
            cx1, cy1, cx2, cy2 = comp["bbox"]
            draw.rectangle([cx1, cy1, cx2, cy2], outline=comp_color, width=2)
            draw.text((cx1, cy1 - 14), comp["type"], fill=comp_color, font=small_font)
    
    return image


def create_ui():
    """Create Gradio interface with full controls."""
    
    with gr.Blocks(title="Utility Asset Detector", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🔌 Utility Asset Detector
        **Open-vocabulary detection using DART/SAM3. Edit the class prompts to detect anything.**
        """)
        
        with gr.Tabs():
            # Tab 1: Simple Detection
            with gr.Tab("🎯 Simple Detection"):
                gr.Markdown("Detect any objects by specifying class names (one per line)")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        simple_image = gr.Image(label="Upload Image", type="numpy", height=350)
                        
                        simple_classes = gr.Textbox(
                            label="Classes to Detect (one per line)",
                            lines=8,
                            value="utility pole\ndisc insulator\nceramic insulator\npower line\ncross arm\ntransformer",
                        )
                        
                        with gr.Row():
                            simple_conf = gr.Slider(0.1, 0.9, value=0.3, step=0.05, label="Confidence")
                            simple_nms = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="NMS Threshold")
                        
                        simple_btn = gr.Button("🔍 Detect", variant="primary")
                    
                    with gr.Column(scale=1):
                        simple_output = gr.Image(label="Results", height=350)
                        simple_summary = gr.Markdown(label="Summary")
                        simple_json = gr.Code(label="JSON", language="json", lines=10)
                
                simple_btn.click(
                    detect_with_classes,
                    inputs=[simple_image, simple_classes, simple_conf, simple_nms],
                    outputs=[simple_output, simple_json, simple_summary],
                )
            
            # Tab 2: Hierarchical Detection
            with gr.Tab("🏗️ Hierarchical Detection"):
                gr.Markdown("Three-level detection: Structures → Components → Conditions")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        hier_image = gr.Image(label="Upload Image", type="numpy", height=300)
                        
                        with gr.Accordion("Structure Classes", open=True):
                            hier_structures = gr.Textbox(lines=5, value=DEFAULT_STRUCTURES, show_label=False)
                            hier_struct_conf = gr.Slider(0.1, 0.9, value=0.5, label="Structure Confidence")
                        
                        with gr.Accordion("Component Classes", open=True):
                            hier_components = gr.Textbox(lines=6, value=DEFAULT_COMPONENTS, show_label=False)
                            hier_comp_conf = gr.Slider(0.1, 0.9, value=0.4, label="Component Confidence")
                        
                        with gr.Accordion("Condition Classes", open=False):
                            hier_conditions = gr.Textbox(lines=5, value=DEFAULT_CONDITIONS, show_label=False)
                            hier_cond_conf = gr.Slider(0.1, 0.9, value=0.5, label="Condition Confidence")
                        
                        hier_nms = gr.Slider(0.1, 1.0, value=0.4, label="NMS Threshold (lower = fewer overlaps)")
                        hier_btn = gr.Button("🔍 Run Hierarchical Detection", variant="primary")
                    
                    with gr.Column(scale=1):
                        hier_output = gr.Image(label="Results", height=350)
                        hier_summary = gr.Markdown(label="Summary")
                        hier_json = gr.Code(label="JSON", language="json", lines=12)
                
                hier_btn.click(
                    run_hierarchical,
                    inputs=[
                        hier_image, hier_structures, hier_components, hier_conditions,
                        hier_struct_conf, hier_comp_conf, hier_cond_conf, hier_nms
                    ],
                    outputs=[hier_output, hier_json, hier_summary],
                )
        
        gr.Markdown("""
        ---
        **Tips:**
        - Use specific class names: "disc insulator" works better than "insulator"
        - Lower NMS threshold removes more overlapping detections
        - Adjust confidence thresholds based on results
        
        [GitHub](https://github.com/menonpg/utility-asset-detector) | [DART](https://github.com/mkturkcan/DART)
        """)
    
    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
