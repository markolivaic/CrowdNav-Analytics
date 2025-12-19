"""Configuration module for CrowdNav-Analytics pedestrian flow system.

This module centralizes all hyperparameters, model configuration, heatmap
math parameters, and file system paths. Using a centralized configuration
ensures consistency across perception, analytics, and visualization components
and simplifies deployment across different environments.
"""

from pathlib import Path
from typing import List, Tuple

# File System Paths
# BASE_DIR resolves to project root by traversing one directory level
# (from src/config.py -> project_root/)
BASE_DIR: Path = Path(__file__).resolve().parent.parent
INPUT_DIR: Path = BASE_DIR / 'data' / 'input'
OUTPUT_DIR: Path = BASE_DIR / 'data' / 'output'

# Perception Configuration
# YOLOv8x selected for maximum accuracy in aerial footage where pedestrians
# appear small (20-50 pixels). The extra-large variant provides better recall
# at the cost of inference speed, which is acceptable for batch processing.
MODEL_WEIGHTS: str = 'yolov8x.pt'
INFERENCE_SIZE: int = 1920
CONFIDENCE_THRESHOLD: float = 0.1
IOU_THRESHOLD: float = 0.5
CLASS_ID_PERSON: int = 0

# SAHI Sliced Inference Configuration
# SAHI improves small object detection by processing overlapping tiles.
# For aerial footage, this yields 2-3x more detections compared to
# full-frame inference at the cost of processing time.
SAHI_ENABLED: bool = True
SAHI_SLICE_SIZE: int = 640
SAHI_OVERLAP_RATIO: float = 0.2
SAHI_POSTPROCESS_TYPE: str = "NMS"
SAHI_POSTPROCESS_MATCH_THRESHOLD: float = 0.5

# Heatmap Analytics Configuration
# Decay rate of 0.95 means 5% heat loss per frame. At 30 FPS, a stationary
# point loses 50% intensity after ~14 frames (~0.5 seconds), creating
# visible trails without excessive ghosting.
HEATMAP_DECAY_RATE: float = 0.95

# Sigma controls Gaussian blur spread. Value of 20 creates smooth gradients
# that blend overlapping pedestrian paths while maintaining spatial resolution.
HEATMAP_SIGMA: float = 20.0

# Brush radius determines the heat deposit area per detection. Value of 15
# pixels ensures sufficient data survives the Gaussian blur operation.
HEATMAP_BRUSH_RADIUS: int = 15

# Intensity step is the heat value added per frame per detection.
# Capped at 1.0 to prevent saturation in high-density areas.
HEATMAP_INTENSITY_STEP: float = 1.0

# Custom Gradient LUT Control Points (Blue -> Cyan -> Yellow -> Red)
# Each tuple: (normalized_position, BGR_color)
# Custom LUT avoids cv2.COLORMAP_HOT white saturation at high values,
# preserving color distinction across the full intensity range.
# Position 0.0 is black (transparent), 1.0 is maximum density.
LUT_CONTROL_POINTS: List[Tuple[float, Tuple[int, int, int]]] = [
    (0.0,  (0, 0, 0)),        # Black (transparent region)
    (0.2,  (255, 0, 0)),      # Blue (low density)
    (0.5,  (255, 255, 0)),    # Cyan (moderate density)
    (0.8,  (0, 255, 255)),    # Yellow (high density)
    (1.0,  (0, 0, 255)),      # Red (maximum density)
]

# Visualization Configuration
# Opacity of 1.0 for additive blending. Since black pixels contribute
# nothing in additive mode, only colored regions affect the output.
HEATMAP_OPACITY: float = 1.0

# Masking threshold (0-255). Values below this are set to black.
# 5% threshold (13/255) removes low-intensity noise while preserving
# legitimate heat trails.
HEATMAP_MASK_THRESHOLD: int = 13

# Dashboard and Annotation Colors (BGR format)
COLOR_TEXT: Tuple[int, int, int] = (255, 255, 255)
COLOR_HEADER: Tuple[int, int, int] = (0, 255, 255)
COLOR_WARN: Tuple[int, int, int] = (0, 0, 255)
COLOR_OK: Tuple[int, int, int] = (0, 255, 0)
COLOR_MODERATE: Tuple[int, int, int] = (0, 255, 255)

# OpenCV font constant
FONT: int = 0

# Crowd Density Thresholds
# These thresholds categorize crowd density for operational dashboard.
# Values based on typical urban pedestrian flow patterns.
DENSITY_LOW_THRESHOLD: int = 50
DENSITY_HIGH_THRESHOLD: int = 120
