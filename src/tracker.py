"""Pedestrian detection wrapper using YOLOv8 with SAHI sliced inference.

This module wraps YOLOv8 object detection with optional SAHI (Slicing Aided
Hyper Inference) for improved small object detection in aerial footage.
SAHI processes overlapping image tiles independently, then merges results
using NMS, which significantly improves recall for distant pedestrians.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from ultralytics import YOLO

from src.config import (
    CLASS_ID_PERSON,
    CONFIDENCE_THRESHOLD,
    INFERENCE_SIZE,
    SAHI_ENABLED,
    SAHI_SLICE_SIZE,
    SAHI_OVERLAP_RATIO,
    SAHI_POSTPROCESS_TYPE,
    SAHI_POSTPROCESS_MATCH_THRESHOLD
)

logger = logging.getLogger(__name__)


@dataclass
class BoxWrapper:
    """Wrapper to provide ultralytics-compatible box interface for SAHI results.

    SAHI returns predictions in its own format. This wrapper provides the
    same interface as ultralytics Results.boxes, allowing the downstream
    pipeline to process both SAHI and standard YOLO outputs identically.

    Attributes:
        xyxy: Tensor of bounding boxes in (x1, y1, x2, y2) format.
        conf: Tensor of confidence scores for each detection.
    """

    xyxy: torch.Tensor
    conf: torch.Tensor

    def __len__(self) -> int:
        """Return number of detections."""
        return len(self.xyxy)

    def __iter__(self):
        """Iterate over individual box items."""
        for i in range(len(self.xyxy)):
            yield BoxItem(self.xyxy[i:i + 1], self.conf[i:i + 1])


@dataclass
class BoxItem:
    """Single detection box for iteration compatibility.

    Attributes:
        xyxy: Tensor containing single bounding box coordinates.
        conf: Tensor containing single confidence score.
    """

    xyxy: torch.Tensor
    conf: torch.Tensor


@dataclass
class SAHIResultWrapper:
    """Wrapper to provide ultralytics Results-compatible interface for SAHI output.

    Converts SAHI prediction format to match ultralytics Results structure,
    enabling seamless integration with the existing visualization and
    analytics pipeline without requiring format-specific handling.

    Attributes:
        boxes: BoxWrapper containing detections, or None if no detections.
    """

    boxes: Optional[BoxWrapper]

    def __init__(self, boxes_xyxy: np.ndarray, confidences: np.ndarray) -> None:
        """Initialize wrapper from numpy arrays.

        Args:
            boxes_xyxy: Numpy array of shape (N, 4) with bounding box
                coordinates in (x1, y1, x2, y2) format.
            confidences: Numpy array of shape (N,) with confidence scores.
        """
        if len(boxes_xyxy) > 0:
            self.boxes = BoxWrapper(
                xyxy=torch.tensor(boxes_xyxy, dtype=torch.float32),
                conf=torch.tensor(confidences, dtype=torch.float32)
            )
        else:
            self.boxes = None


class PedestrianTracker:
    """Wraps YOLOv8 detection with optional SAHI sliced inference.

    For aerial footage where pedestrians appear as 20-50 pixel objects,
    standard YOLO inference often misses detections due to the small
    object size relative to the model's receptive field. SAHI addresses
    this by processing overlapping 640x640 tiles, effectively "zooming in"
    on different regions of the frame.

    Performance characteristics:
    - SAHI mode: ~5-10 FPS, 2-3x more detections
    - Standard mode: ~30 FPS, baseline detection rate

    The tradeoff favors SAHI for batch processing where detection accuracy
    matters more than real-time performance.

    Attributes:
        device: Compute device ('cuda' or 'cpu').
        model: YOLOv8 model instance.
        sahi_enabled: Whether SAHI sliced inference is active.
        sahi_model: SAHI detection model (if SAHI is enabled).
    """

    def __init__(self, model_path: str) -> None:
        """Initialize detector with YOLOv8 model and optional SAHI.

        Automatically selects CUDA if available. SAHI initialization
        is attempted if enabled in config; falls back to standard
        inference if SAHI import fails or initialization errors occur.

        Args:
            model_path: Path to YOLOv8 weights file (.pt format).

        Raises:
            Exception: If YOLOv8 model loading fails.
        """
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sahi_enabled: bool = SAHI_ENABLED

        try:
            self.model = YOLO(model_path)
            logger.info(f"YOLO initialized on {self.device} with {model_path}")

            if self.sahi_enabled:
                self._init_sahi(model_path)

        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise

    def _init_sahi(self, model_path: str) -> None:
        """Initialize SAHI detection model for sliced inference.

        SAHI requires separate model initialization from the AutoDetectionModel
        factory. If initialization fails, the tracker falls back to standard
        YOLO inference rather than failing completely.

        Args:
            model_path: Path to YOLOv8 weights file.
        """
        try:
            from sahi import AutoDetectionModel

            self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model_path=model_path,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                device=self.device
            )
            logger.info(
                f"SAHI enabled: slice={SAHI_SLICE_SIZE}px, "
                f"overlap={SAHI_OVERLAP_RATIO * 100:.0f}%"
            )
        except ImportError:
            logger.warning("SAHI not installed, falling back to standard inference")
            self.sahi_enabled = False
        except Exception as e:
            logger.warning(f"SAHI initialization failed: {e}, using standard inference")
            self.sahi_enabled = False

    def track(self, frame: np.ndarray) -> SAHIResultWrapper:
        """Run detection on frame using configured inference mode.

        Routes to either SAHI sliced inference or standard YOLO inference
        based on configuration. Both paths return SAHIResultWrapper for
        consistent downstream processing.

        Args:
            frame: Input BGR video frame as numpy array.

        Returns:
            SAHIResultWrapper containing detection results with boxes
            and confidence scores.
        """
        if self.sahi_enabled:
            return self._track_sahi(frame)
        else:
            return self._track_standard(frame)

    def _track_sahi(self, frame: np.ndarray) -> SAHIResultWrapper:
        """Run SAHI sliced inference for improved small object detection.

        Divides the frame into overlapping tiles (default 640x640 with 20%
        overlap), runs YOLO inference on each tile, then merges results
        using Non-Maximum Suppression. This approach yields 2-3x more
        detections on aerial footage compared to full-frame inference.

        Args:
            frame: Input BGR video frame as numpy array.

        Returns:
            SAHIResultWrapper containing merged detection results.
        """
        from sahi.predict import get_sliced_prediction

        result = get_sliced_prediction(
            frame,
            self.sahi_model,
            slice_height=SAHI_SLICE_SIZE,
            slice_width=SAHI_SLICE_SIZE,
            overlap_height_ratio=SAHI_OVERLAP_RATIO,
            overlap_width_ratio=SAHI_OVERLAP_RATIO,
            postprocess_type=SAHI_POSTPROCESS_TYPE,
            postprocess_match_threshold=SAHI_POSTPROCESS_MATCH_THRESHOLD,
            verbose=0
        )

        # Filter for person class and extract coordinates
        boxes_list = []
        conf_list = []

        for pred in result.object_prediction_list:
            if pred.category.id == CLASS_ID_PERSON:
                bbox = pred.bbox.to_xyxy()
                boxes_list.append([bbox[0], bbox[1], bbox[2], bbox[3]])
                conf_list.append(pred.score.value)

        boxes_array = np.array(boxes_list) if boxes_list else np.array([])
        conf_array = np.array(conf_list) if conf_list else np.array([])

        return SAHIResultWrapper(boxes_array, conf_array)

    def _track_standard(self, frame: np.ndarray) -> SAHIResultWrapper:
        """Run standard YOLO inference on full frame.

        Fallback mode when SAHI is disabled or unavailable. Processes
        the entire frame in a single inference pass, which is faster
        but may miss small objects in aerial footage.

        Args:
            frame: Input BGR video frame as numpy array.

        Returns:
            SAHIResultWrapper containing detection results.
        """
        results = self.model.predict(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            classes=[CLASS_ID_PERSON],
            device=self.device,
            verbose=False,
            imgsz=INFERENCE_SIZE
        )

        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            boxes_array = result.boxes.xyxy.cpu().numpy()
            conf_array = result.boxes.conf.cpu().numpy()
        else:
            boxes_array = np.array([])
            conf_array = np.array([])

        return SAHIResultWrapper(boxes_array, conf_array)
