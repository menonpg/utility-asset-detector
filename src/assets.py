"""
Asset hierarchy definitions for T&D utility infrastructure.

Defines the three-level detection hierarchy:
1. Structures: Poles, towers, substations
2. Components: Insulators, transformers, cross-arms
3. Conditions: Damage indicators, vegetation, wear
"""

from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class Asset:
    """Base class for all utility assets."""
    id: str
    type: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2]
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "confidence": self.confidence,
            "bbox": self.bbox,
        }


@dataclass
class Condition:
    """Condition assessment for an asset."""
    status: str  # good, attention_needed, damaged, critical
    issues: list[str] = field(default_factory=list)
    severity: str = "none"  # none, low, moderate, high
    details: list[dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "issues": self.issues,
            "severity": self.severity,
        }


@dataclass
class Component(Asset):
    """Level 2: Components attached to structures."""
    condition: Condition = field(default_factory=lambda: Condition(status="unknown"))
    parent_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        d = super().to_dict()
        d["condition"] = self.condition.to_dict()
        if self.parent_id:
            d["parent_id"] = self.parent_id
        return d


@dataclass  
class Structure(Asset):
    """Level 1: Primary structures (poles, towers, substations)."""
    subtype: Optional[str] = None
    components: list[Component] = field(default_factory=list)
    condition: Condition = field(default_factory=lambda: Condition(status="unknown"))
    
    @property
    def has_damage(self) -> bool:
        """Check if structure or any component has damage."""
        if self.condition.severity in ["moderate", "high"]:
            return True
        return any(c.condition.severity in ["moderate", "high"] for c in self.components)
    
    def to_dict(self) -> dict:
        d = super().to_dict()
        if self.subtype:
            d["subtype"] = self.subtype
        d["components"] = [c.to_dict() for c in self.components]
        d["condition"] = self.condition.to_dict()
        return d


class AssetHierarchy:
    """
    Manages the asset class hierarchy and provides class lists for detection.
    
    Default hierarchy:
    
    Structures:
      - utility_pole (wood, concrete, steel)
      - transmission_tower (lattice, monopole, H-frame)
      - distribution_pole
      - substation
      - distribution_box
      
    Components:
      - insulator (pin, suspension, strain, post)
      - transformer (pole-mounted, pad-mounted)
      - cross_arm
      - conductor / power_line
      - guy_wire
      - anchor
      - switch
      - fuse
      - lightning_arrester
      - meter
      - capacitor_bank
      
    Conditions:
      - crack
      - break
      - rust / corrosion
      - burn_mark
      - hole
      - woodpecker_damage
      - lean / tilt
      - missing_part
      - vegetation_contact
      - bird_nest
      - oil_leak
    """
    
    # Default class definitions
    DEFAULT_STRUCTURES = [
        "utility pole",
        "wood pole", 
        "concrete pole",
        "steel pole",
        "transmission tower",
        "lattice tower",
        "monopole tower",
        "H-frame tower",
        "distribution pole",
        "substation",
        "distribution box",
        "junction box",
    ]
    
    DEFAULT_COMPONENTS = [
        "insulator",
        "pin insulator",
        "suspension insulator", 
        "strain insulator",
        "post insulator",
        "transformer",
        "pole-mounted transformer",
        "pad-mounted transformer",
        "cross arm",
        "conductor",
        "power line",
        "guy wire",
        "anchor",
        "switch",
        "fuse",
        "cutout fuse",
        "lightning arrester",
        "surge arrester",
        "meter",
        "capacitor bank",
        "recloser",
        "voltage regulator",
    ]
    
    DEFAULT_CONDITIONS = [
        "crack",
        "break",
        "fracture",
        "rust",
        "corrosion",
        "burn mark",
        "flashover damage",
        "hole",
        "woodpecker hole",
        "woodpecker damage",
        "lean",
        "severe lean",
        "tilt",
        "missing part",
        "missing insulator",
        "broken insulator",
        "vegetation contact",
        "tree contact",
        "overgrown vegetation",
        "bird nest",
        "animal damage",
        "oil leak",
        "oil stain",
        "damaged wire",
        "sagging wire",
        "broken conductor",
    ]
    
    def __init__(
        self,
        structures: Optional[list[str]] = None,
        components: Optional[list[str]] = None,
        conditions: Optional[list[str]] = None,
    ):
        self.structures = structures or self.DEFAULT_STRUCTURES.copy()
        self.components = components or self.DEFAULT_COMPONENTS.copy()
        self.conditions = conditions or self.DEFAULT_CONDITIONS.copy()
        
    @classmethod
    def from_yaml(cls, path: str) -> "AssetHierarchy":
        """Load hierarchy from YAML config file."""
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            # Return defaults if config not found
            return cls()
            
        hierarchy_data = data.get("hierarchy", {})
        
        return cls(
            structures=hierarchy_data.get("structures"),
            components=hierarchy_data.get("components"),
            conditions=hierarchy_data.get("conditions"),
        )
    
    def get_structure_classes(self) -> list[str]:
        """Get list of structure class names for detection."""
        return self.structures
    
    def get_component_classes(self) -> list[str]:
        """Get list of component class names for detection."""
        return self.components
    
    def get_condition_classes(self) -> list[str]:
        """Get list of condition/damage class names for detection."""
        return self.conditions
    
    def get_all_classes(self) -> list[str]:
        """Get all classes combined."""
        return self.structures + self.components + self.conditions
    
    def add_structure(self, name: str) -> None:
        """Add a custom structure class."""
        if name not in self.structures:
            self.structures.append(name)
            
    def add_component(self, name: str) -> None:
        """Add a custom component class."""
        if name not in self.components:
            self.components.append(name)
            
    def add_condition(self, name: str) -> None:
        """Add a custom condition class."""
        if name not in self.conditions:
            self.conditions.append(name)


# Predefined hierarchies for common use cases
TRANSMISSION_HIERARCHY = AssetHierarchy(
    structures=[
        "transmission tower",
        "lattice tower", 
        "monopole tower",
        "H-frame tower",
        "dead-end tower",
        "angle tower",
        "suspension tower",
    ],
    components=[
        "suspension insulator",
        "strain insulator",
        "conductor",
        "ground wire",
        "shield wire",
        "spacer",
        "vibration damper",
        "armor rod",
        "warning sphere",
    ],
    conditions=[
        "rust",
        "corrosion",
        "broken insulator",
        "damaged conductor",
        "missing hardware",
        "bird nest",
        "vegetation encroachment",
        "foundation damage",
    ],
)


DISTRIBUTION_HIERARCHY = AssetHierarchy(
    structures=[
        "utility pole",
        "wood pole",
        "concrete pole",
        "distribution pole",
        "junction pole",
    ],
    components=[
        "transformer",
        "pole-mounted transformer",
        "insulator",
        "pin insulator",
        "cross arm",
        "fuse",
        "cutout fuse",
        "lightning arrester",
        "guy wire",
        "anchor",
        "meter",
        "service drop",
    ],
    conditions=[
        "woodpecker hole",
        "woodpecker damage",
        "rot",
        "lean",
        "severe lean",
        "crack",
        "broken cross arm",
        "missing insulator",
        "oil leak",
        "vegetation contact",
        "damaged wire",
    ],
)


SUBSTATION_HIERARCHY = AssetHierarchy(
    structures=[
        "substation",
        "transformer bay",
        "control building",
        "switchgear",
        "capacitor bank",
    ],
    components=[
        "power transformer",
        "circuit breaker",
        "disconnect switch",
        "current transformer",
        "potential transformer",
        "lightning arrester",
        "busbar",
        "insulator",
        "control cabinet",
    ],
    conditions=[
        "oil leak",
        "rust",
        "corrosion",
        "burn mark",
        "vegetation encroachment",
        "damaged bushing",
        "broken insulator",
        "ground fault evidence",
    ],
)
