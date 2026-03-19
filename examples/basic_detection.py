#!/usr/bin/env python3
"""
Basic example: Detect utility assets in a single image.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector import UtilityAssetDetector


def main():
    # Initialize detector
    detector = UtilityAssetDetector(
        config="configs/assets.yaml",
        device="cuda",
        model_variant="full"  # Use "repvit" for faster inference
    )
    
    # Run detection
    result = detector.detect("examples/sample_pole.jpg")
    
    # Print results
    print(f"Detected {result.total_structures} structures")
    print(f"Total components: {result.total_components}")
    print(f"Priority: {result.priority}")
    print()
    
    for structure in result.structures:
        print(f"📍 {structure.type} (confidence: {structure.confidence:.2f})")
        print(f"   Status: {structure.condition.status}")
        if structure.condition.issues:
            print(f"   Issues: {', '.join(structure.condition.issues)}")
            
        for comp in structure.components:
            status_emoji = {
                "good": "✅",
                "attention_needed": "⚠️",
                "damaged": "🔶",
                "critical": "🔴"
            }.get(comp.condition.status, "❓")
            
            print(f"   {status_emoji} {comp.type}: {comp.condition.status}")
            
    # Save JSON output
    output_path = "examples/output/detection_result.json"
    Path(output_path).parent.mkdir(exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(result.to_json())
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
