#!/usr/bin/env python3
"""
Video inspection: Process drone footage with tracking.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector import UtilityAssetDetector
from src.results import InspectionReport
import json


def main():
    # Initialize detector with faster variant for video
    detector = UtilityAssetDetector(
        config="configs/assets.yaml",
        device="cuda",
        model_variant="pruned"  # Faster for real-time
    )
    
    video_path = "examples/drone_inspection.mp4"
    output_dir = Path("examples/output")
    output_dir.mkdir(exist_ok=True)
    
    # Process video
    print(f"Processing {video_path}...")
    
    results = []
    damage_frames = []
    
    for frame_result in detector.process_video(
        video_path,
        track=True,
        output_path=str(output_dir / "annotated_video.mp4")
    ):
        results.append(frame_result.detection)
        
        # Log frames with damage
        if frame_result.detection.has_damage:
            damage_frames.append({
                "frame": frame_result.frame_id,
                "timestamp": f"{frame_result.timestamp:.2f}s",
                "priority": frame_result.detection.priority,
                "damaged_assets": frame_result.detection.get_damaged_assets()
            })
            
        # Progress indicator
        if frame_result.frame_id % 100 == 0:
            print(f"  Frame {frame_result.frame_id}...")
    
    # Generate report
    report = InspectionReport(results=results)
    
    print(f"\n📊 Inspection Summary")
    print(f"   Total frames: {report.total_images}")
    print(f"   Structures detected: {report.total_structures}")
    print(f"   Components detected: {report.total_components}")
    print(f"   Frames with damage: {report.images_with_damage}")
    print(f"   Damage rate: {report.images_with_damage / max(1, report.total_images) * 100:.1f}%")
    
    # Damage breakdown
    damage_summary = report.get_damage_summary()
    if damage_summary:
        print(f"\n⚠️ Damage Types Found:")
        for damage_type, count in list(damage_summary.items())[:10]:
            print(f"   - {damage_type}: {count} occurrences")
    
    # Save outputs
    with open(output_dir / "damage_log.json", "w") as f:
        json.dump(damage_frames, f, indent=2)
        
    with open(output_dir / "full_report.json", "w") as f:
        f.write(report.to_json())
        
    print(f"\n✅ Outputs saved to {output_dir}/")
    print(f"   - annotated_video.mp4")
    print(f"   - damage_log.json")
    print(f"   - full_report.json")


if __name__ == "__main__":
    main()
