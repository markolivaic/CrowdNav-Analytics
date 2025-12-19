"""Main video processing pipeline for CrowdNav-Analytics.

This module handles the complete pedestrian flow analysis pipeline from
video ingestion through detection, heatmap accumulation, and annotated
output generation. It serves as the entry point for batch video processing
with configurable frame limits and resource management.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from src.config import INPUT_DIR, MODEL_WEIGHTS, OUTPUT_DIR
from src.tracker import PedestrianTracker
from src.analytics import FlowDynamicsEngine
from src.visualizer import AnalyticsVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def run_pipeline(filename: str, frame_limit: Optional[int] = None) -> None:
    """Run the pedestrian flow analysis pipeline on a video file.

    Processes video frames through three stages:
    1. Perception: YOLOv8 + SAHI detection for pedestrian localization
    2. Analytics: Temporal heatmap accumulation with decay and smoothing
    3. Visualization: Custom LUT colormap and dashboard rendering

    The pipeline uses try-finally to ensure video resources are released
    even if processing is interrupted or encounters errors. This prevents
    file handle leaks and ensures output videos have valid headers.

    Args:
        filename: Name of the video file in data/input/ directory.
        frame_limit: Maximum frames to process. If None, processes all
            frames. Useful for testing on long videos.
    """
    input_path: Path = INPUT_DIR / filename
    output_path: Path = OUTPUT_DIR / f"analyzed_{filename}"

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    # Component initialization
    try:
        tracker = PedestrianTracker(MODEL_WEIGHTS)
        visualizer = AnalyticsVisualizer()
        logger.info("Pipeline components initialized")
    except Exception as e:
        logger.critical(f"Initialization failed: {e}")
        return

    # Video capture setup
    video_capture = cv2.VideoCapture(str(input_path))
    if not video_capture.isOpened():
        logger.error("Failed to open video file")
        return

    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Apply frame limit if specified
    frames_to_process = total_frames
    if frame_limit is not None and frame_limit < total_frames:
        frames_to_process = frame_limit
        logger.info(f"Frame limit applied: processing {frames_to_process} of {total_frames} frames")

    # Initialize analytics engine with video dimensions
    analytics_engine = FlowDynamicsEngine(width, height)

    # Video writer setup
    # mp4v codec provides broad compatibility across platforms
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        str(output_path), fourcc, fps, (width, height)
    )

    logger.info(f"Processing: {filename} [{width}x{height} @ {fps:.1f} FPS]")

    try:
        for frame_idx in tqdm(range(frames_to_process), desc="Processing", unit="frames"):
            ret, frame = video_capture.read()
            if not ret:
                logger.warning(f"Frame read failed at index {frame_idx}")
                break

            # Stage 1: Perception
            tracks = tracker.track(frame)

            # Stage 2: Analytics
            # Extract bounding boxes for heatmap accumulation
            if tracks.boxes is not None and len(tracks.boxes) > 0:
                xyxy = tracks.boxes.xyxy
                current_boxes = xyxy.cpu().numpy() if hasattr(xyxy, 'cpu') else xyxy
            else:
                current_boxes = np.array([])

            heat_map = analytics_engine.update(current_boxes)

            # Stage 3: Visualization
            output_frame = visualizer.render(frame, tracks, heat_map)
            video_writer.write(output_frame)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
    finally:
        # Resource cleanup ensures output video has valid headers
        # and file handles are released for subsequent processing
        video_capture.release()
        video_writer.release()
        logger.info(f"Output saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CrowdNav-Analytics: Pedestrian Flow Analysis Pipeline"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Filename of the video in data/input/"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum frames to process (default: all frames)"
    )

    args = parser.parse_args()

    # Ensure data directories exist
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_pipeline(args.video, args.limit)
