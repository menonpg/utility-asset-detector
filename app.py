#!/usr/bin/env python3
"""
Gradio Web UI for Utility Asset Detector.

Launch: python app.py
One-click: Open in Colab with GPU runtime
"""

import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Check if running in limited environment (no DART yet)
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

if not DEMO_MODE:
    try:
        from src.detector import UtilityAssetDetector
        from src.assets import AssetHierarchy, TRANSMISSION_HIERARCHY, DISTRIBUTION_HIERARCHY
        DETECTOR_AVAILABLE = True
    except ImportError:
        DETECTOR_AVAILABLE = False
        print("⚠️ DART not installed. Run: cd DART && pip install -e .")
else:
    DETECTOR_AVAILABLE = False

# Global detector instance (lazy loaded)
_detector = None
_current_config = None


def get_detector(hierarchy_type: str = "general", device: str = "cuda"):
    """Get or create detector with specified hierarchy."""
    global _detector, _current_config
    
    if not DETECTOR_AVAILABLE:
        return None
        
    config_key = f"{hierarchy_type}_{device}"
    
    if _detector is None or _current_config != config_key:
        print(f"Loading detector: {hierarchy_type} on {device}...")
        _detector = UtilityAssetDetector(
            config="configs/assets.yaml",
            device=device,
            model_variant="full"
        )
        
        # Apply hierarchy preset
        if hierarchy_type == "transmission":
            _detector.hierarchy = TRANSMISSION_HIERARCHY
        elif hierarchy_type == "distribution":
            _detector.hierarchy = DISTRIBUTION_HIERARCHY
        # else: use general/default
        
        _detector._structure_classes = _detector.hierarchy.get_structure_classes()
        _detector._component_classes = _detector.hierarchy.get_component_classes()
        _detector._condition_classes = _detector.hierarchy.get_condition_classes()
        
        _current_config = config_key
        print("✅ Detector ready")
        
    return _detector


def draw_detections(image: Image.Image, result) -> Image.Image:
    """Draw bounding boxes and labels on image."""
    draw = ImageDraw.Draw(image)
    
    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    # Colors by severity
    colors = {
        "good": "#00FF00",
        "attention_needed": "#FFFF00", 
        "damaged": "#FFA500",
        "critical": "#FF0000",
        "unknown": "#808080",
    }
    
    for structure in result.structures:
        x1, y1, x2, y2 = [int(c) for c in structure.bbox]
        color = colors.get(structure.condition.status, "#FFFFFF")
        
        # Draw structure box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Label background
        label = f"{structure.type} ({structure.confidence:.0%})"
        bbox = draw.textbbox((x1, y1 - 20), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 20), label, fill="black", font=font)
        
        # Draw component boxes
        for comp in structure.components:
            cx1, cy1, cx2, cy2 = [int(c) for c in comp.bbox]
            comp_color = colors.get(comp.condition.status, "#CCCCCC")
            draw.rectangle([cx1, cy1, cx2, cy2], outline=comp_color, width=2)
            
            # Component label
            comp_label = f"{comp.type}"
            draw.text((cx1, cy1 - 15), comp_label, fill=comp_color, font=small_font)
    
    return image


def detect_assets(
    image: np.ndarray,
    hierarchy_type: str,
    structure_conf: float,
    component_conf: float,
    condition_conf: float,
) -> tuple[np.ndarray, str, str]:
    """
    Run detection on uploaded image.
    
    Returns: (annotated_image, json_results, summary_text)
    """
    if image is None:
        return None, "{}", "Please upload an image"
    
    # Convert to PIL
    pil_image = Image.fromarray(image).convert("RGB")
    
    if not DETECTOR_AVAILABLE:
        # Demo mode - return placeholder
        return (
            image,
            json.dumps({"error": "DART not installed", "demo_mode": True}, indent=2),
            "⚠️ **Demo Mode**\n\nDART is not installed. To enable detection:\n\n"
            "```bash\ngit clone https://github.com/mkturkcan/DART.git\n"
            "cd DART && pip install -e .\n```\n\n"
            "Or use the Colab notebook for one-click setup with free GPU."
        )
    
    # Get detector
    detector = get_detector(hierarchy_type)
    
    # Update confidence thresholds
    detector.config.structure_confidence = structure_conf
    detector.config.component_confidence = component_conf
    detector.config.condition_confidence = condition_conf
    
    # Run detection
    result = detector.detect(pil_image)
    
    # Draw annotations
    annotated = draw_detections(pil_image.copy(), result)
    
    # Build summary
    summary_parts = [
        f"## Detection Results\n",
        f"**Structures found:** {result.total_structures}",
        f"**Components found:** {result.total_components}",
        f"**Priority:** {result.priority}",
        f"**Damage detected:** {'Yes ⚠️' if result.has_damage else 'No ✅'}",
        "",
    ]
    
    for struct in result.structures:
        status_emoji = {
            "good": "✅",
            "attention_needed": "⚠️",
            "damaged": "🔶",
            "critical": "🔴",
        }.get(struct.condition.status, "❓")
        
        summary_parts.append(f"### {status_emoji} {struct.type}")
        summary_parts.append(f"- Confidence: {struct.confidence:.0%}")
        summary_parts.append(f"- Status: {struct.condition.status}")
        
        if struct.condition.issues:
            summary_parts.append(f"- Issues: {', '.join(struct.condition.issues)}")
            
        if struct.components:
            summary_parts.append(f"- Components: {len(struct.components)}")
            for comp in struct.components:
                comp_emoji = {
                    "good": "✅", "damaged": "🔶", "critical": "🔴"
                }.get(comp.condition.status, "•")
                summary_parts.append(f"  - {comp_emoji} {comp.type}: {comp.condition.status}")
        
        summary_parts.append("")
    
    summary = "\n".join(summary_parts)
    json_output = result.to_json()
    
    return np.array(annotated), json_output, summary


def create_ui():
    """Create Gradio interface."""
    
    with gr.Blocks(
        title="Utility Asset Detector",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("""
        # 🔌 Utility Asset Detector
        
        Real-time detection of T&D utility infrastructure using [DART](https://github.com/mkturkcan/DART).
        
        **Detects:** Poles, towers, insulators, transformers, cross-arms, damage, and more.
        
        Upload an image of utility infrastructure to analyze.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Image",
                    type="numpy",
                    height=400,
                )
                
                hierarchy_type = gr.Radio(
                    choices=["general", "transmission", "distribution"],
                    value="general",
                    label="Asset Hierarchy",
                    info="Choose preset for your infrastructure type"
                )
                
                with gr.Accordion("Detection Settings", open=False):
                    structure_conf = gr.Slider(
                        0.1, 0.9, value=0.4, step=0.05,
                        label="Structure Confidence",
                        info="Minimum confidence for poles/towers"
                    )
                    component_conf = gr.Slider(
                        0.1, 0.9, value=0.35, step=0.05,
                        label="Component Confidence", 
                        info="Minimum confidence for insulators/transformers"
                    )
                    condition_conf = gr.Slider(
                        0.1, 0.9, value=0.3, step=0.05,
                        label="Condition Confidence",
                        info="Minimum confidence for damage detection"
                    )
                
                detect_btn = gr.Button("🔍 Detect Assets", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Detection Results",
                    type="numpy",
                    height=400,
                )
                
                with gr.Tabs():
                    with gr.Tab("Summary"):
                        summary_output = gr.Markdown(
                            label="Detection Summary",
                            value="Upload an image and click Detect to see results."
                        )
                    with gr.Tab("JSON"):
                        json_output = gr.Code(
                            label="JSON Output",
                            language="json",
                            lines=20,
                        )
        
        # Example images removed - Wikipedia blocks Gradio downloads
        # Upload your own utility pole images to test
        
        # Connect button
        detect_btn.click(
            fn=detect_assets,
            inputs=[
                input_image,
                hierarchy_type,
                structure_conf,
                component_conf,
                condition_conf,
            ],
            outputs=[output_image, json_output, summary_output],
        )
        
        gr.Markdown("""
        ---
        **Links:** [GitHub](https://github.com/menonpg/utility-asset-detector) | 
        [DART](https://github.com/mkturkcan/DART) |
        [Blog Post](https://menonlab-blog-production.up.railway.app/dart-real-time-object-detection-sam3)
        
        Built with [DART](https://github.com/mkturkcan/DART) and [Gradio](https://gradio.app)
        """)
    
    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set True for public link
    )
