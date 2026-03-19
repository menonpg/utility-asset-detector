# Utility Asset Detector

Real-time detection of transmission and distribution (T&D) utility assets using [DART](https://github.com/mkturkcan/DART) — a training-free open-vocabulary object detector built on SAM3.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/menonpg/utility-asset-detector/blob/main/notebooks/colab_demo.ipynb)

## Features

- **Zero Training Required** — Detect poles, insulators, transformers, and more without labeled data
- **Hierarchical Detection** — First detect structures (poles, towers), then components (insulators, cross-arms)
- **Damage Assessment** — Identify visual damage patterns (cracks, rust, leaning, missing parts)
- **Video Pipeline** — Process inspection footage with tracking
- **Web UI** — Gradio interface for easy testing
- **Configurable Classes** — YAML-based asset hierarchy, easy to extend

## 🚀 One-Click Launch (No Setup Required)

**Option 1: Google Colab (Free GPU)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/menonpg/utility-asset-detector/blob/main/notebooks/colab_demo.ipynb)

Click the badge above → Runtime → Run all → Use the web UI that appears.

**Option 2: Local with Web UI**

```bash
# Clone repos
git clone https://github.com/menonpg/utility-asset-detector.git
git clone https://github.com/mkturkcan/DART.git

# Install
cd DART && pip install -e . && cd ..
cd utility-asset-detector
pip install -r requirements.txt

# Launch Web UI
python app.py
# Opens at http://localhost:7860
```

## Quick Start (CLI)

```bash
# Run detection on an image
python src/detector.py --image inspection_photo.jpg --output results/

# Run on video with tracking
python src/detector.py --video drone_footage.mp4 --output results/ --track
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Input Image/Video                     │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Level 1: Structure Detection                │
│         poles, towers, substations, buildings            │
└─────────────────────────────────────────────────────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  Utility Pole │  │ Transmission  │  │  Substation   │
│               │  │    Tower      │  │               │
└───────────────┘  └───────────────┘  └───────────────┘
            │              │              │
            ▼              ▼              ▼
┌─────────────────────────────────────────────────────────┐
│            Level 2: Component Detection                  │
│   insulators, transformers, cross-arms, conductors       │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│            Level 3: Condition Assessment                 │
│     damage, corrosion, vegetation, missing parts         │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    Structured Output                     │
│              JSON with hierarchy + damage                │
└─────────────────────────────────────────────────────────┘
```

## Asset Hierarchy

The detector uses a three-level hierarchy defined in `configs/assets.yaml`:

### Level 1: Structures
- Utility pole (wood, concrete, steel)
- Transmission tower (lattice, monopole)
- Substation
- Distribution box

### Level 2: Components (detected within structure ROI)
- Insulators (pin, suspension, strain)
- Transformers (pole-mounted, pad-mounted)
- Cross-arms
- Conductors / power lines
- Guy wires and anchors
- Switches and fuses
- Lightning arresters

### Level 3: Conditions
- Physical damage (cracks, breaks, holes)
- Corrosion / rust
- Leaning / misalignment
- Vegetation encroachment
- Missing components
- Burn marks / flashover evidence
- Wildlife damage (woodpecker holes, nests)

## Usage

### Basic Detection

```python
from src.detector import UtilityAssetDetector

detector = UtilityAssetDetector(
    config="configs/assets.yaml",
    device="cuda"
)

# Single image
results = detector.detect("inspection_photo.jpg")
print(results.to_json())

# Access hierarchical results
for structure in results.structures:
    print(f"Found {structure.type} at {structure.bbox}")
    for component in structure.components:
        print(f"  - {component.type}: {component.condition}")
```

### Video Processing

```python
from src.detector import UtilityAssetDetector

detector = UtilityAssetDetector(config="configs/assets.yaml")

# Process video with tracking
for frame_result in detector.process_video("drone_footage.mp4", track=True):
    # Each frame has structure → component → condition hierarchy
    for structure in frame_result.structures:
        if structure.has_damage:
            print(f"Frame {frame_result.frame_id}: Damage on {structure.type}")
```

### Damage Report Generation

```python
from src.detector import UtilityAssetDetector
from src.reporting import generate_inspection_report

detector = UtilityAssetDetector(config="configs/assets.yaml")
results = detector.detect_batch("inspection_images/")

# Generate PDF report
report = generate_inspection_report(
    results,
    output="inspection_report.pdf",
    include_images=True
)
```

## Configuration

Edit `configs/assets.yaml` to customize:

```yaml
structures:
  utility_pole:
    subtypes: ["wood pole", "concrete pole", "steel pole"]
    components:
      - insulator
      - transformer
      - cross_arm
      - guy_wire
    
  transmission_tower:
    subtypes: ["lattice tower", "monopole tower", "H-frame"]
    components:
      - suspension_insulator
      - strain_insulator
      - conductor
      - ground_wire

conditions:
  damage_indicators:
    - "crack"
    - "break"
    - "rust"
    - "corrosion"
    - "burn mark"
    - "hole"
    - "leaning"
    - "missing part"
    
  vegetation_risk:
    - "tree contact"
    - "vegetation encroachment"
    - "overgrown brush"
```

## Performance

| Resolution | Structures/sec | Full Hierarchy/sec | GPU |
|------------|----------------|-------------------|-----|
| 1008px | 15.8 | 8.2 | RTX 4080 |
| 720px | 24.3 | 12.1 | RTX 4080 |
| 1008px | 8.5 | 4.2 | RTX 3060 |

Hierarchical detection runs component detection on each structure ROI, so throughput scales with structure count.

## Output Format

```json
{
  "image": "pole_inspection_001.jpg",
  "timestamp": "2026-03-19T10:30:00Z",
  "structures": [
    {
      "id": "structure_0",
      "type": "utility_pole",
      "subtype": "wood pole",
      "confidence": 0.94,
      "bbox": [120, 50, 280, 650],
      "components": [
        {
          "type": "transformer",
          "confidence": 0.91,
          "bbox": [150, 120, 250, 220],
          "condition": {
            "status": "damaged",
            "issues": ["rust", "oil leak stain"],
            "severity": "moderate"
          }
        },
        {
          "type": "insulator",
          "confidence": 0.88,
          "bbox": [180, 80, 220, 120],
          "condition": {
            "status": "good",
            "issues": [],
            "severity": "none"
          }
        }
      ],
      "condition": {
        "status": "attention_needed",
        "issues": ["woodpecker holes", "slight lean"],
        "severity": "low"
      }
    }
  ],
  "summary": {
    "total_structures": 1,
    "total_components": 2,
    "damage_found": true,
    "priority": "routine"
  }
}
```

## Integration Examples

### With Inspection Workflow

```python
from src.detector import UtilityAssetDetector
from src.integrations import upload_to_gis, create_work_order

detector = UtilityAssetDetector(config="configs/assets.yaml")

# Process field photos
for photo in inspection_batch:
    results = detector.detect(photo, gps_coords=photo.gps)
    
    # Upload to GIS with damage annotations
    upload_to_gis(results, layer="utility_inspections")
    
    # Create work orders for damaged assets
    if results.priority in ["urgent", "high"]:
        create_work_order(results)
```

### Drone Pipeline

```python
from src.detector import UtilityAssetDetector

detector = UtilityAssetDetector(
    config="configs/assets.yaml",
    model_variant="pruned"  # Faster for real-time drone feed
)

# Real-time detection during flight
for frame in drone.video_stream():
    results = detector.detect(frame)
    
    # Highlight damage for operator
    if results.has_critical_damage:
        drone.mark_poi(frame.gps, priority="high")
        drone.capture_detail_photo()
```

## Web UI

The Gradio web interface provides:

- **Drag & drop** image upload
- **Live visualization** with bounding boxes
- **Hierarchy presets** (General, Transmission, Distribution)
- **Adjustable thresholds** for fine-tuning
- **JSON export** of detection results

```bash
# Launch locally
python app.py

# With public URL (for sharing)
python app.py --share
```

![Web UI Screenshot](docs/webui-screenshot.png)

## Requirements

- Python 3.11+
- PyTorch 2.7+ with CUDA (or use Colab for free GPU)
- DART (auto-installed)
- See `requirements.txt` for full list

## License

Apache 2.0

## References

- [DART: Detect Anything in Real Time](https://github.com/mkturkcan/DART)
- [SAM3: Segment Anything Model 3](https://github.com/facebookresearch/sam3)
- [The Menon Lab Blog Post](https://menonlab-blog-production.up.railway.app/dart-real-time-object-detection-sam3)
