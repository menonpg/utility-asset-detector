#!/usr/bin/env python3
"""
Example: GIS integration with geotagged inspection photos.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from dataclasses import dataclass
from typing import Optional

from src.detector import UtilityAssetDetector
from src.results import InspectionReport


@dataclass
class GeotaggedPhoto:
    """Photo with GPS metadata."""
    path: str
    lat: float
    lon: float
    altitude: Optional[float] = None
    heading: Optional[float] = None


def extract_gps_from_exif(image_path: str) -> Optional[tuple[float, float]]:
    """Extract GPS coordinates from image EXIF data."""
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS, GPSTAGS
        
        image = Image.open(image_path)
        exif = image._getexif()
        
        if not exif:
            return None
            
        gps_info = {}
        for tag_id, value in exif.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "GPSInfo":
                for gps_tag_id, gps_value in value.items():
                    gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_info[gps_tag] = gps_value
                    
        if not gps_info:
            return None
            
        def convert_to_degrees(value):
            d, m, s = value
            return d + (m / 60.0) + (s / 3600.0)
            
        lat = convert_to_degrees(gps_info.get("GPSLatitude", (0, 0, 0)))
        if gps_info.get("GPSLatitudeRef", "N") == "S":
            lat = -lat
            
        lon = convert_to_degrees(gps_info.get("GPSLongitude", (0, 0, 0)))
        if gps_info.get("GPSLongitudeRef", "E") == "W":
            lon = -lon
            
        return (lat, lon)
        
    except Exception:
        return None


def generate_geojson(results: list, output_path: str):
    """Generate GeoJSON file from detection results."""
    features = []
    
    for result in results:
        if not result.gps_coords:
            continue
            
        lat, lon = result.gps_coords
        
        for structure in result.structures:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]  # GeoJSON uses [lon, lat]
                },
                "properties": {
                    "asset_type": structure.type,
                    "confidence": structure.confidence,
                    "condition_status": structure.condition.status,
                    "severity": structure.condition.severity,
                    "issues": structure.condition.issues,
                    "component_count": len(structure.components),
                    "image": result.image_path,
                    "priority": result.priority,
                }
            }
            features.append(feature)
            
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)
        
    return geojson


def main():
    """Process geotagged inspection photos."""
    
    detector = UtilityAssetDetector(
        config="configs/assets.yaml",
        device="cuda"
    )
    
    # Simulate geotagged photos (in practice, extract from EXIF)
    photos = [
        GeotaggedPhoto("inspection/pole_001.jpg", 40.7128, -74.0060),
        GeotaggedPhoto("inspection/pole_002.jpg", 40.7129, -74.0061),
        GeotaggedPhoto("inspection/pole_003.jpg", 40.7130, -74.0062),
    ]
    
    print("🌍 Processing geotagged inspection photos...")
    
    results = []
    for photo in photos:
        # Try to extract GPS from EXIF if not provided
        gps = (photo.lat, photo.lon)
        if gps == (0, 0):
            extracted = extract_gps_from_exif(photo.path)
            if extracted:
                gps = extracted
                
        # Run detection with GPS
        result = detector.detect(photo.path, gps_coords=gps)
        results.append(result)
        
        print(f"  📍 {photo.path}: {result.total_structures} structures at ({gps[0]:.4f}, {gps[1]:.4f})")
        
    # Generate outputs
    output_dir = Path("examples/output/gis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # GeoJSON for GIS software (QGIS, ArcGIS, etc.)
    geojson = generate_geojson(results, str(output_dir / "detections.geojson"))
    print(f"\n✅ GeoJSON saved: {len(geojson['features'])} features")
    
    # Damage-only GeoJSON for priority view
    damage_results = [r for r in results if r.has_damage]
    if damage_results:
        generate_geojson(damage_results, str(output_dir / "damage_only.geojson"))
        print(f"✅ Damage GeoJSON saved: {len(damage_results)} locations")
        
    # Full report
    report = InspectionReport(results=results)
    with open(output_dir / "inspection_report.json", "w") as f:
        f.write(report.to_json())
        
    print(f"\n📊 Summary:")
    print(f"   Total photos: {report.total_images}")
    print(f"   Structures found: {report.total_structures}")
    print(f"   Locations with damage: {report.images_with_damage}")


if __name__ == "__main__":
    main()
