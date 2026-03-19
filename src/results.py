"""
Result classes for detection output.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .assets import Structure, Component, Condition


@dataclass
class DetectionResult:
    """Result from detecting assets in a single image."""
    
    structures: list[Structure] = field(default_factory=list)
    image_path: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    gps_coords: Optional[tuple[float, float]] = None
    
    @property
    def total_structures(self) -> int:
        return len(self.structures)
    
    @property
    def total_components(self) -> int:
        return sum(len(s.components) for s in self.structures)
    
    @property
    def has_damage(self) -> bool:
        return any(s.has_damage for s in self.structures)
    
    @property
    def priority(self) -> str:
        """Determine inspection priority based on damage."""
        severities = []
        for s in self.structures:
            severities.append(s.condition.severity)
            for c in s.components:
                severities.append(c.condition.severity)
                
        if "high" in severities:
            return "urgent"
        elif "moderate" in severities:
            return "high"
        elif "low" in severities:
            return "routine"
        return "normal"
    
    def to_dict(self) -> dict:
        return {
            "image": self.image_path,
            "timestamp": self.timestamp,
            "gps": {"lat": self.gps_coords[0], "lon": self.gps_coords[1]} 
                   if self.gps_coords else None,
            "structures": [s.to_dict() for s in self.structures],
            "summary": {
                "total_structures": self.total_structures,
                "total_components": self.total_components,
                "damage_found": self.has_damage,
                "priority": self.priority,
            }
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    def get_damaged_assets(self) -> list[dict]:
        """Get list of all assets with damage."""
        damaged = []
        for s in self.structures:
            if s.condition.severity in ["moderate", "high"]:
                damaged.append({
                    "type": "structure",
                    "asset": s.type,
                    "condition": s.condition.to_dict(),
                    "bbox": s.bbox,
                })
            for c in s.components:
                if c.condition.severity in ["moderate", "high"]:
                    damaged.append({
                        "type": "component",
                        "asset": c.type,
                        "parent": s.type,
                        "condition": c.condition.to_dict(),
                        "bbox": c.bbox,
                    })
        return damaged


@dataclass
class FrameResult:
    """Result from processing a single video frame."""
    
    frame_id: int
    timestamp: float  # seconds from video start
    detection: DetectionResult
    track_ids: dict = field(default_factory=dict)  # asset_id -> track_id mapping
    
    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "detection": self.detection.to_dict(),
            "track_ids": self.track_ids,
        }


@dataclass
class InspectionReport:
    """Aggregated report from multiple images/frames."""
    
    results: list[DetectionResult] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    
    @property
    def total_images(self) -> int:
        return len(self.results)
    
    @property
    def total_structures(self) -> int:
        return sum(r.total_structures for r in self.results)
    
    @property
    def total_components(self) -> int:
        return sum(r.total_components for r in self.results)
    
    @property
    def images_with_damage(self) -> int:
        return sum(1 for r in self.results if r.has_damage)
    
    def get_all_damaged_assets(self) -> list[dict]:
        """Get all damaged assets across all results."""
        all_damaged = []
        for i, result in enumerate(self.results):
            for asset in result.get_damaged_assets():
                asset["image_index"] = i
                asset["image_path"] = result.image_path
                all_damaged.append(asset)
        return all_damaged
    
    def get_damage_summary(self) -> dict:
        """Get summary of damage types found."""
        damage_counts = {}
        for result in self.results:
            for s in result.structures:
                for issue in s.condition.issues:
                    damage_counts[issue] = damage_counts.get(issue, 0) + 1
                for c in s.components:
                    for issue in c.condition.issues:
                        damage_counts[issue] = damage_counts.get(issue, 0) + 1
        return dict(sorted(damage_counts.items(), key=lambda x: -x[1]))
    
    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "summary": {
                "total_images": self.total_images,
                "total_structures": self.total_structures,
                "total_components": self.total_components,
                "images_with_damage": self.images_with_damage,
                "damage_rate": self.images_with_damage / max(1, self.total_images),
            },
            "damage_summary": self.get_damage_summary(),
            "damaged_assets": self.get_all_damaged_assets(),
            "results": [r.to_dict() for r in self.results],
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
