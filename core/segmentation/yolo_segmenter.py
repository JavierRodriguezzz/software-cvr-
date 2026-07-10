"""CervixAI V2 — YOLO segmentation backend (stub).

This class satisfies the ISegmenter interface but is NOT yet
implemented.  It serves as the architectural placeholder that allows
the backend to be swapped by changing ``config.yaml`` from
``model_backend: "unet"`` to ``model_backend: "yolo"`` — without
modifying any pipeline or core logic.

Implementation will be added in a future sprint once YOLO segmentation
masks have been validated against the U-Net baseline.
"""

from __future__ import annotations

import logging

import numpy as np

from core.domain import SegmentationResult
from core.interfaces import ISegmenter

logger = logging.getLogger(__name__)


class YOLOSegmenter(ISegmenter):
    """Stub ISegmenter for YOLO-based cervical segmentation.

    Args:
        model_path: Path to the YOLO ``.pt`` weights file (not loaded yet).
        imgsz: Inference image size.
        conf: Confidence threshold.
    """

    def __init__(self, model_path: str, imgsz: int = 640, conf: float = 0.25) -> None:
        """Record configuration without loading the model."""
        self._model_path = model_path
        self._imgsz = imgsz
        self._conf = conf
        logger.warning(
            "YOLOSegmenter is a stub — inference is not implemented yet. "
            "Switch config.yaml to model_backend: 'unet' for production use."
        )

    @property
    def backend_name(self) -> str:
        """Return the backend identifier."""
        return "yolo"

    def predict(self, image_bgr: np.ndarray) -> SegmentationResult:
        """Not implemented.

        Args:
            image_bgr: Input image (unused).

        Raises:
            NotImplementedError: Always raised until this backend is
                implemented.
        """
        raise NotImplementedError(
            "YOLOSegmenter.predict() is not yet implemented. "
            "Use model_backend: 'unet' in config.yaml."
        )
