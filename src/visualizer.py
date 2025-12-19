"""Rendering module for pedestrian flow heatmap visualization.

This module handles frame annotation, custom colormap application, and
dashboard overlay rendering. Uses a custom lookup table (LUT) instead of
OpenCV's built-in colormaps to provide better color distinction across
the full intensity range and transparent black regions.
"""

import cv2
import numpy as np
from ultralytics.engine.results import Results

from src.config import (
    COLOR_HEADER,
    COLOR_TEXT,
    COLOR_WARN,
    COLOR_OK,
    COLOR_MODERATE,
    FONT,
    HEATMAP_OPACITY,
    HEATMAP_MASK_THRESHOLD,
    LUT_CONTROL_POINTS,
    DENSITY_LOW_THRESHOLD,
    DENSITY_HIGH_THRESHOLD
)


def _build_custom_lut() -> np.ndarray:
    """Build 256-entry BGR lookup table from control points.

    Creates a smooth gradient by linearly interpolating between control
    points defined in config. Custom LUT provides several advantages over
    cv2.applyColorMap:

    1. Black (0,0,0) at position 0 ensures low-intensity regions remain
       transparent in additive blending mode
    2. Blue->Cyan->Yellow->Red gradient provides intuitive thermal encoding
    3. Avoids white saturation at high values that occurs with COLORMAP_HOT

    Returns:
        Numpy array of shape (256, 3) with uint8 BGR values for each
        intensity level 0-255.
    """
    lut = np.zeros((256, 3), dtype=np.uint8)

    for i in range(len(LUT_CONTROL_POINTS) - 1):
        pos_start, color_start = LUT_CONTROL_POINTS[i]
        pos_end, color_end = LUT_CONTROL_POINTS[i + 1]

        idx_start = int(pos_start * 255)
        idx_end = int(pos_end * 255)

        if idx_end <= idx_start:
            continue

        for idx in range(idx_start, idx_end + 1):
            # Linear interpolation between control points
            t = (idx - idx_start) / (idx_end - idx_start)
            for c in range(3):
                lut[idx, c] = int(color_start[c] + t * (color_end[c] - color_start[c]))

    return lut


# Pre-compute LUT at module load to avoid per-frame computation
_CUSTOM_LUT: np.ndarray = _build_custom_lut()


class AnalyticsVisualizer:
    """Handles compositing of heatmap overlay and operational dashboard.

    Renders the temporal density heatmap onto video frames using custom
    colormap and additive blending. The black regions in the LUT ensure
    areas with no pedestrian activity remain unchanged from the original
    video, creating a clean overlay effect.
    """

    @staticmethod
    def render(frame: np.ndarray, tracks: Results, heat_map: np.ndarray) -> np.ndarray:
        """Render heatmap overlay and dashboard onto video frame.

        Pipeline:
        1. Dynamic normalization based on actual max value
        2. Custom LUT application for Blue->Cyan->Yellow->Red gradient
        3. Masking of low-intensity noise
        4. Additive blending with original frame
        5. Detection bounding boxes (subtle gray)
        6. Dashboard overlay with crowd metrics

        Args:
            frame: Input BGR video frame as numpy array.
            tracks: Detection results containing bounding boxes.
            heat_map: Float32 heat matrix from analytics engine.

        Returns:
            Annotated frame with heatmap overlay and dashboard.
        """
        # Dynamic normalization based on actual maximum value
        # This ensures the full color range is used regardless of
        # absolute heat values, which vary based on crowd density.
        max_val = heat_map.max()
        if max_val > 0:
            heatmap_norm = heat_map / max_val
        else:
            heatmap_norm = heat_map

        heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)

        # Apply custom LUT for Blue->Cyan->Yellow->Red gradient
        # LUT indexing: each pixel value (0-255) maps to a BGR color
        heatmap_color = _CUSTOM_LUT[heatmap_uint8]

        # Mask low-intensity values to remove noise and ensure black
        # regions remain truly black for clean additive blending.
        # Threshold of ~5% (13/255) removes sensor noise while
        # preserving legitimate heat trails.
        low_val_mask = heatmap_uint8 < HEATMAP_MASK_THRESHOLD
        heatmap_color[low_val_mask] = 0

        # Additive blending: original frame + colored heatmap
        # Black pixels (0,0,0) contribute nothing, so only areas with
        # detected activity receive the colored overlay.
        output = cv2.addWeighted(
            frame, 1.0,
            heatmap_color, HEATMAP_OPACITY,
            0
        )

        # Draw detection bounding boxes (subtle gray to avoid distraction)
        if tracks.boxes is not None:
            for box in tracks.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(output, (x1, y1), (x2, y2), (200, 200, 200), 1)

        # Render operational dashboard
        person_count = len(tracks.boxes) if tracks.boxes else 0
        AnalyticsVisualizer._render_dashboard(output, person_count)

        return output

    @staticmethod
    def _render_dashboard(frame: np.ndarray, count: int) -> None:
        """Render semi-transparent operational dashboard overlay.

        Creates a header overlay displaying real-time crowd analytics
        including active target count and density classification. Uses
        alpha blending (60% overlay, 40% original) to maintain visibility
        of underlying heatmap while providing clear operational metrics.

        Density thresholds are calibrated for typical urban pedestrian
        scenarios:
        - LOW: < 50 targets (normal flow)
        - MODERATE: 50-120 targets (busy but manageable)
        - HIGH: > 120 targets (potential congestion)

        Args:
            frame: Frame to render dashboard on (modified in-place).
            count: Current pedestrian count from detection results.
        """
        # Semi-transparent overlay background
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (350, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Density classification based on configured thresholds
        if count < DENSITY_LOW_THRESHOLD:
            status = "LOW ACTIVITY"
            color = COLOR_OK
        elif count < DENSITY_HIGH_THRESHOLD:
            status = "MODERATE"
            color = COLOR_MODERATE
        else:
            status = "HIGH DENSITY"
            color = COLOR_WARN

        # Dashboard text rendering
        cv2.putText(frame, "CROWD ANALYTICS", (35, 45), FONT, 0.5, COLOR_HEADER, 1)
        cv2.putText(frame, f"Targets: {count}", (35, 70), FONT, 0.8, COLOR_TEXT, 2)
        cv2.putText(frame, f"Status: {status}", (35, 95), FONT, 0.6, color, 1)
