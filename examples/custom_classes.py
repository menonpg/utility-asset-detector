#!/usr/bin/env python3
"""
Example: Customize detection classes for specific use case.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector import UtilityAssetDetector
from src.assets import AssetHierarchy, TRANSMISSION_HIERARCHY, DISTRIBUTION_HIERARCHY


def transmission_inspection():
    """Use predefined transmission line hierarchy."""
    print("🔌 Transmission Line Inspection")
    print("-" * 40)
    
    # Use transmission-specific classes
    detector = UtilityAssetDetector(
        config="configs/assets.yaml",
        device="cuda"
    )
    
    # Override with transmission hierarchy
    detector.hierarchy = TRANSMISSION_HIERARCHY
    detector._structure_classes = detector.hierarchy.get_structure_classes()
    detector._component_classes = detector.hierarchy.get_component_classes()
    detector._condition_classes = detector.hierarchy.get_condition_classes()
    
    print(f"Structures: {detector._structure_classes}")
    print(f"Components: {detector._component_classes[:5]}...")
    print()


def distribution_inspection():
    """Use predefined distribution line hierarchy."""
    print("🏠 Distribution Line Inspection")
    print("-" * 40)
    
    detector = UtilityAssetDetector(
        config="configs/assets.yaml",
        device="cuda"
    )
    
    detector.hierarchy = DISTRIBUTION_HIERARCHY
    detector._structure_classes = detector.hierarchy.get_structure_classes()
    detector._component_classes = detector.hierarchy.get_component_classes()
    detector._condition_classes = detector.hierarchy.get_condition_classes()
    
    print(f"Structures: {detector._structure_classes}")
    print(f"Components: {detector._component_classes[:5]}...")
    print()


def custom_hierarchy():
    """Create a fully custom hierarchy."""
    print("🔧 Custom Hierarchy Example")
    print("-" * 40)
    
    # Define custom classes for solar + wind inspection
    custom = AssetHierarchy(
        structures=[
            "solar panel array",
            "solar tracker",
            "wind turbine",
            "wind turbine tower",
            "inverter station",
            "combiner box",
        ],
        components=[
            "solar panel",
            "mounting bracket",
            "junction box",
            "inverter",
            "turbine blade",
            "nacelle",
            "cable tray",
        ],
        conditions=[
            "crack",
            "delamination",
            "hot spot",
            "discoloration",
            "corrosion",
            "bird droppings",
            "vegetation shading",
            "broken panel",
            "loose connection",
            "blade damage",
            "erosion",
        ]
    )
    
    detector = UtilityAssetDetector(
        config="configs/assets.yaml",
        device="cuda"
    )
    
    detector.hierarchy = custom
    detector._structure_classes = custom.get_structure_classes()
    detector._component_classes = custom.get_component_classes()
    detector._condition_classes = custom.get_condition_classes()
    
    print(f"Structures: {custom.structures}")
    print(f"Components: {custom.components}")
    print(f"Conditions: {custom.conditions[:5]}...")
    print()
    
    # Now use detector.detect() as normal


def add_custom_classes():
    """Add individual classes to existing hierarchy."""
    print("➕ Adding Custom Classes")
    print("-" * 40)
    
    hierarchy = AssetHierarchy()  # Start with defaults
    
    # Add custom classes
    hierarchy.add_structure("fiber optic pole")
    hierarchy.add_structure("communication tower")
    hierarchy.add_component("fiber splice enclosure")
    hierarchy.add_component("antenna")
    hierarchy.add_condition("fiber break")
    hierarchy.add_condition("antenna misalignment")
    
    print("Added:")
    print(f"  Structures: fiber optic pole, communication tower")
    print(f"  Components: fiber splice enclosure, antenna")
    print(f"  Conditions: fiber break, antenna misalignment")


if __name__ == "__main__":
    transmission_inspection()
    distribution_inspection()
    custom_hierarchy()
    add_custom_classes()
