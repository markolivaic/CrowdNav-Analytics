"""Temporal heatmap analytics engine for pedestrian flow analysis.

This module handles the mathematical operations for accumulating and
processing pedestrian density data over time. The heat matrix uses float32
precision to prevent truncation errors during exponential decay operations
that would cause temporal inconsistencies in the visualization.
"""

import logging
from typing import Union

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from src.config import (
    HEATMAP_DECAY_RATE,
    HEATMAP_INTENSITY_STEP,
    HEATMAP_SIGMA,
    HEATMAP_BRUSH_RADIUS
)

logger = logging.getLogger(__name__)


class FlowDynamicsEngine:
    """Calculates temporal density accumulation for pedestrian flow visualization.

    This engine maintains a float32 heat matrix that accumulates pedestrian
    positions over time. The matrix implements exponential decay to create
    temporal trails and Gaussian smoothing to produce spatially coherent
    density gradients.

    Float32 precision is critical: decay operations (M *= 0.95) on uint8
    matrices cause precision loss that manifests as temporal jitter and
    incorrect trail lengths. Float32 maintains sub-pixel precision across
    hundreds of decay iterations.

    Attributes:
        heat_matrix: Float32 numpy array storing accumulated heat values.
            Shape matches input video dimensions (height, width).
    """

    def __init__(self, width: int, height: int) -> None:
        """Initialize the heat matrix with video dimensions.

        Args:
            width: Video frame width in pixels.
            height: Video frame height in pixels.
        """
        self.heat_matrix: np.ndarray = np.zeros((height, width), dtype=np.float32)
        logger.debug(f"Heat matrix initialized: {width}x{height} float32")

    def update(self, detections: Union[np.ndarray, None]) -> np.ndarray:
        """Update heat matrix with new detections and return smoothed result.

        Implements the temporal heatmap pipeline:
        1. Decay: M_t = M_{t-1} * lambda (exponential decay for trail effect)
        2. Accumulation: Add heat at detected pedestrian foot positions
        3. Clipping: Prevent saturation in high-density areas
        4. Smoothing: H = M * G(sigma) for spatial coherence

        The foot position (bottom-center of bounding box) is used instead of
        centroid because it represents the actual ground contact point,
        producing more accurate flow paths on the ground plane.

        Args:
            detections: Numpy array of bounding boxes with shape (N, 4+)
                where each row contains [x1, y1, x2, y2, ...]. Can be
                empty array or None if no detections in current frame.

        Returns:
            Smoothed heat matrix as float32 numpy array with values in
            range [0, 1]. The caller handles normalization for display.
        """
        # Decay: M_t = M_{t-1} * lambda
        # Exponential decay creates temporal trails. At 0.95 decay and 30 FPS,
        # a point reaches 50% intensity after ~14 frames (~0.5 seconds).
        self.heat_matrix *= HEATMAP_DECAY_RATE

        # Accumulation: Add heat at foot positions using circular brush
        if isinstance(detections, np.ndarray) and detections.size > 0:
            for box in detections:
                try:
                    x1, y1, x2, y2 = map(int, box[:4])
                    cx = int((x1 + x2) / 2)
                    cy = int(y2)  # Foot position (bottom of bounding box)

                    # Circular brush deposits heat over an area rather than
                    # single pixel. This ensures the Gaussian blur has
                    # sufficient data to create visible gradients.
                    cv2.circle(
                        self.heat_matrix,
                        (cx, cy),
                        HEATMAP_BRUSH_RADIUS,
                        HEATMAP_INTENSITY_STEP,
                        -1  # Filled circle
                    )
                except (ValueError, IndexError):
                    continue

        # Clipping prevents saturation in areas with sustained high density
        self.heat_matrix = np.clip(self.heat_matrix, 0.0, 1.0)

        logger.debug(f"Heat matrix max intensity: {self.heat_matrix.max():.4f}")

        # Smoothing: H = M * G(sigma)
        # scipy.ndimage.gaussian_filter operates on float32 directly,
        # avoiding precision loss from uint8 conversion that cv2.GaussianBlur
        # would require for equivalent operation.
        smoothed: np.ndarray = gaussian_filter(
            self.heat_matrix,
            sigma=HEATMAP_SIGMA
        )

        return smoothed
