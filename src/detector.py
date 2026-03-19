#!/usr/bin/env python3
"""
Utility Asset Detector - Hierarchical T&D infrastructure detection using DART.

Usage:
    python detector.py --image inspection.jpg --output results/
    python detector.py --video drone.mp4 --output results/ --track
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import torch
import yaml
from PIL import Image

# DART imports (installed from mkturkcan/DART)
from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast

from .assets import AssetHierarchy
from .results import DetectionResult, FrameResult, Structure, Component, Condition


@dataclass
class DetectorConfig:
    """Configuration for the utility asset detector."""
    
    # Model settings
    checkpoint: str = "sam3.pt"
    device: str = "cuda"
    model_variant: str = "full"  # full, pruned, repvit
    
    # Detection settings
    structure_confidence: float = 0.4
    component_confidence: float = 0.35
    condition_confidence: float = 0.3
    
    # Hierarchical settings
    roi_padding: float = 0.1  # Expand structure bbox for component detection
    
    # Performance
    resolution: int = 1008
    use_trt: bool = False
    trt_backbone: Optional[str] = None
    trt_enc_dec: Optional[str] = None


class UtilityAssetDetector:
    """
    Hierarchical detector for T&D utility assets.
    
    Detection flow:
    1. Detect structures (poles, towers, substations)
    2. For each structure, detect components within ROI
    3. For each component, assess condition/damage
    """
    
    def __init__(
        self,
        config: str | dict | DetectorConfig = "configs/assets.yaml",
        device: str = "cuda",
        model_variant: str = "full",
    ):
        """
        Initialize the detector.
        
        Args:
            config: Path to YAML config, dict, or DetectorConfig object
            device: Torch device ('cuda' or 'cpu')
            model_variant: Model variant ('full', 'pruned', 'repvit')
        """
        # Load configuration
        if isinstance(config, str):
            self.config = self._load_yaml_config(config)
        elif isinstance(config, dict):
            self.config = DetectorConfig(**config)
        else:
            self.config = config
            
        self.config.device = device
        self.config.model_variant = model_variant
        
        # Load asset hierarchy
        self.hierarchy = AssetHierarchy.from_yaml(
            config if isinstance(config, str) else "configs/assets.yaml"
        )
        
        # Initialize DART predictor
        self.predictor = self._init_predictor()
        
        # Pre-compute class lists for each detection level
        self._structure_classes = self.hierarchy.get_structure_classes()
        self._component_classes = self.hierarchy.get_component_classes()
        self._condition_classes = self.hierarchy.get_condition_classes()
        
    def _load_yaml_config(self, path: str) -> DetectorConfig:
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return DetectorConfig(**data.get("detector", {}))
    
    def _init_predictor(self) -> Sam3MultiClassPredictorFast:
        """Initialize the DART predictor based on model variant."""
        if self.config.model_variant == "full":
            predictor = Sam3MultiClassPredictorFast.from_pretrained(
                self.config.checkpoint,
                device=self.config.device
            )
        elif self.config.model_variant == "pruned":
            # Use pruned backbone (faster, slightly lower accuracy)
            from sam3.distillation.sam3_student import build_sam3_student_model
            model = build_sam3_student_model(
                backbone_config="pruned_16",
                teacher_checkpoint=self.config.checkpoint,
                device=self.config.device,
            )
            predictor = Sam3MultiClassPredictorFast(model, device=self.config.device)
        elif self.config.model_variant == "repvit":
            # Use lightweight student backbone (fastest)
            from sam3.distillation.sam3_student import build_sam3_student_model
            model = build_sam3_student_model(
                backbone_config="repvit_m2_3",
                teacher_checkpoint=self.config.checkpoint,
                device=self.config.device,
            )
            predictor = Sam3MultiClassPredictorFast(model, device=self.config.device)
        else:
            raise ValueError(f"Unknown model variant: {self.config.model_variant}")
            
        return predictor
    
    def detect(
        self,
        image: str | Image.Image,
        gps_coords: Optional[tuple[float, float]] = None,
    ) -> DetectionResult:
        """
        Run hierarchical detection on a single image.
        
        Args:
            image: Image path or PIL Image
            gps_coords: Optional (lat, lon) coordinates
            
        Returns:
            DetectionResult with hierarchical structure
        """
        # Load image if path
        if isinstance(image, str):
            image_path = image
            image = Image.open(image).convert("RGB")
        else:
            image_path = None
            
        # Level 1: Detect structures
        structures = self._detect_structures(image)
        
        # Level 2: Detect components within each structure
        for structure in structures:
            components = self._detect_components(image, structure.bbox)
            structure.components = components
            
            # Level 3: Assess condition for structure and components
            structure.condition = self._assess_condition(image, structure.bbox, "structure")
            for component in components:
                component.condition = self._assess_condition(
                    image, component.bbox, component.type
                )
        
        # Build result
        result = DetectionResult(
            image_path=image_path,
            structures=structures,
            gps_coords=gps_coords,
        )
        
        return result
    
    def _detect_structures(self, image: Image.Image) -> list[Structure]:
        """Detect Level 1 structures (poles, towers, etc.)."""
        self.predictor.set_classes(self._structure_classes)
        state = self.predictor.set_image(image)
        
        detections = self.predictor.predict(
            state,
            confidence_threshold=self.config.structure_confidence
        )
        
        structures = []
        for i, (box, score, class_id) in enumerate(zip(
            detections['boxes'],
            detections['scores'],
            detections['class_ids']
        )):
            structure = Structure(
                id=f"structure_{i}",
                type=self._structure_classes[class_id],
                confidence=float(score),
                bbox=box.tolist() if hasattr(box, 'tolist') else list(box),
            )
            structures.append(structure)
            
        return structures
    
    def _detect_components(
        self,
        image: Image.Image,
        structure_bbox: list[float],
    ) -> list[Component]:
        """Detect Level 2 components within a structure's ROI."""
        # Expand bbox with padding
        x1, y1, x2, y2 = structure_bbox
        w, h = x2 - x1, y2 - y1
        pad_x = w * self.config.roi_padding
        pad_y = h * self.config.roi_padding
        
        # Clamp to image bounds
        img_w, img_h = image.size
        roi = [
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(img_w, x2 + pad_x),
            min(img_h, y2 + pad_y),
        ]
        
        # Crop ROI
        roi_image = image.crop([int(c) for c in roi])
        
        # Detect components in ROI
        self.predictor.set_classes(self._component_classes)
        state = self.predictor.set_image(roi_image)
        
        detections = self.predictor.predict(
            state,
            confidence_threshold=self.config.component_confidence
        )
        
        components = []
        for i, (box, score, class_id) in enumerate(zip(
            detections['boxes'],
            detections['scores'],
            detections['class_ids']
        )):
            # Convert ROI-relative coords to image coords
            box = box.tolist() if hasattr(box, 'tolist') else list(box)
            box = [
                box[0] + roi[0],
                box[1] + roi[1],
                box[2] + roi[0],
                box[3] + roi[1],
            ]
            
            component = Component(
                id=f"component_{i}",
                type=self._component_classes[class_id],
                confidence=float(score),
                bbox=box,
            )
            components.append(component)
            
        return components
    
    def _assess_condition(
        self,
        image: Image.Image,
        bbox: list[float],
        asset_type: str,
    ) -> Condition:
        """Assess condition/damage for an asset region."""
        # Crop to asset region
        x1, y1, x2, y2 = [int(c) for c in bbox]
        asset_image = image.crop((x1, y1, x2, y2))
        
        # Detect condition indicators
        self.predictor.set_classes(self._condition_classes)
        state = self.predictor.set_image(asset_image)
        
        detections = self.predictor.predict(
            state,
            confidence_threshold=self.config.condition_confidence
        )
        
        # Collect detected issues
        issues = []
        for score, class_id in zip(detections['scores'], detections['class_ids']):
            issue = self._condition_classes[class_id]
            issues.append({
                "type": issue,
                "confidence": float(score)
            })
        
        # Determine severity
        if not issues:
            status = "good"
            severity = "none"
        elif any(i["type"] in ["break", "burn mark", "severe lean"] for i in issues):
            status = "critical"
            severity = "high"
        elif any(i["type"] in ["crack", "rust", "corrosion", "hole"] for i in issues):
            status = "damaged"
            severity = "moderate"
        else:
            status = "attention_needed"
            severity = "low"
            
        return Condition(
            status=status,
            issues=[i["type"] for i in issues],
            severity=severity,
            details=issues,
        )
    
    def process_video(
        self,
        video_path: str,
        track: bool = True,
        output_path: Optional[str] = None,
    ) -> Iterator[FrameResult]:
        """
        Process video with optional tracking.
        
        Args:
            video_path: Path to input video
            track: Enable ByteTrack for persistent IDs
            output_path: Optional path to save annotated video
            
        Yields:
            FrameResult for each processed frame
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        frame_id = 0
        
        # Video writer setup
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Convert BGR to RGB PIL Image
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Run detection
                result = self.detect(image)
                
                # Create frame result
                frame_result = FrameResult(
                    frame_id=frame_id,
                    timestamp=frame_id / cap.get(cv2.CAP_PROP_FPS),
                    detection=result,
                )
                
                # Draw annotations if saving video
                if writer:
                    annotated = self._draw_annotations(frame, result)
                    writer.write(annotated)
                
                yield frame_result
                frame_id += 1
                
        finally:
            cap.release()
            if writer:
                writer.release()
    
    def _draw_annotations(self, frame, result: DetectionResult):
        """Draw detection boxes and labels on frame."""
        import cv2
        
        # Colors by severity
        colors = {
            "good": (0, 255, 0),      # Green
            "attention_needed": (0, 255, 255),  # Yellow
            "damaged": (0, 165, 255),  # Orange
            "critical": (0, 0, 255),   # Red
        }
        
        for structure in result.structures:
            # Draw structure box
            x1, y1, x2, y2 = [int(c) for c in structure.bbox]
            color = colors.get(structure.condition.status, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"{structure.type}: {structure.condition.status}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw component boxes
            for comp in structure.components:
                cx1, cy1, cx2, cy2 = [int(c) for c in comp.bbox]
                comp_color = colors.get(comp.condition.status, (200, 200, 200))
                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), comp_color, 1)
                
        return frame
    
    def detect_batch(
        self,
        image_dir: str,
        output_dir: Optional[str] = None,
    ) -> list[DetectionResult]:
        """Process all images in a directory."""
        results = []
        image_dir = Path(image_dir)
        
        for img_path in sorted(image_dir.glob("*.jpg")) + \
                        sorted(image_dir.glob("*.png")) + \
                        sorted(image_dir.glob("*.jpeg")):
            result = self.detect(str(img_path))
            results.append(result)
            
            if output_dir:
                out_path = Path(output_dir) / f"{img_path.stem}_result.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(result.to_dict(), f, indent=2)
                    
        return results


def main():
    parser = argparse.ArgumentParser(description="Utility Asset Detector")
    parser.add_argument("--image", help="Input image path")
    parser.add_argument("--video", help="Input video path")
    parser.add_argument("--output", "-o", default="results/", help="Output directory")
    parser.add_argument("--config", default="configs/assets.yaml", help="Config file")
    parser.add_argument("--track", action="store_true", help="Enable tracking for video")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--variant", default="full", 
                       choices=["full", "pruned", "repvit"],
                       help="Model variant")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = UtilityAssetDetector(
        config=args.config,
        device=args.device,
        model_variant=args.variant,
    )
    
    os.makedirs(args.output, exist_ok=True)
    
    if args.image:
        # Single image detection
        result = detector.detect(args.image)
        
        # Save JSON result
        out_path = Path(args.output) / f"{Path(args.image).stem}_result.json"
        with open(out_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        print(f"Detected {len(result.structures)} structures")
        print(f"Results saved to {out_path}")
        
        # Print summary
        for struct in result.structures:
            print(f"\n{struct.type} ({struct.confidence:.2f}):")
            print(f"  Condition: {struct.condition.status}")
            if struct.condition.issues:
                print(f"  Issues: {', '.join(struct.condition.issues)}")
            for comp in struct.components:
                print(f"  - {comp.type}: {comp.condition.status}")
                
    elif args.video:
        # Video processing
        video_out = Path(args.output) / f"{Path(args.video).stem}_annotated.mp4"
        
        damage_log = []
        for frame_result in detector.process_video(
            args.video,
            track=args.track,
            output_path=str(video_out)
        ):
            if frame_result.detection.has_damage:
                damage_log.append({
                    "frame": frame_result.frame_id,
                    "timestamp": frame_result.timestamp,
                    "structures": [s.to_dict() for s in frame_result.detection.structures
                                  if s.condition.severity != "none"]
                })
                
        # Save damage log
        log_path = Path(args.output) / f"{Path(args.video).stem}_damage_log.json"
        with open(log_path, "w") as f:
            json.dump(damage_log, f, indent=2)
            
        print(f"Processed video saved to {video_out}")
        print(f"Damage log saved to {log_path}")
        print(f"Found damage in {len(damage_log)} frames")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
