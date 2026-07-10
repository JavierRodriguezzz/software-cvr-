"""CervixAI V2 — Abstract interfaces (Strategy contracts).

All strategy implementations MUST inherit from these ABCs.
The core pipeline depends only on these interfaces, never on
concrete implementations — enabling DI and future backend swaps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

from core.context import PipelineContext
from core.domain import (
    CalibrationResult,
    ConfidenceResult,
    MeasurementResult,
    SegmentationResult,
    VisualizationResult,
)


class ISegmenter(ABC):
    """Contract for segmentation backends (U-Net, YOLO, future models).

    Implementations must be stateless with respect to the pipeline;
    all model state must be encapsulated in the implementation class.
    """

    @abstractmethod
    def predict(self, image_bgr: np.ndarray) -> SegmentationResult:
        """Run inference on a BGR image.

        Args:
            image_bgr: Input image in BGR colour order, shape (H, W, 3).

        Returns:
            SegmentationResult with prediction mask and optional probabilities.
        """
        ...

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return a human-readable name for this segmentation backend."""
        ...


class IMeasurementStrategy(ABC):
    """Contract for canal measurement strategies.

    The strategy encapsulates the algorithm used to determine the
    anatomical axis of the cervical canal and compute its arc-length.
    """

    @abstractmethod
    def measure(
        self,
        prediction: np.ndarray,
        context: PipelineContext,
    ) -> MeasurementResult:
        """Compute the anatomical measurement from a segmentation mask.

        Args:
            prediction: Integer mask (H, W) with values {0, 1, 2}.
            context: The pipeline context for reading configuration and
                writing intermediate artifacts.

        Returns:
            MeasurementResult with the anatomical axis and arc-length.
        """
        ...


class ICalibrationStrategy(ABC):
    """Contract for scale bar detection strategies."""

    @abstractmethod
    def calibrate(self, image_bgr: np.ndarray) -> CalibrationResult:
        """Detect the scale bar and return mm/pixel conversion.

        Args:
            image_bgr: Full original image in BGR.

        Returns:
            CalibrationResult with mm_per_pixel (None on failure).
        """
        ...


class IVisualizationStrategy(ABC):
    """Contract for visualization strategies."""

    @abstractmethod
    def render(self, context: PipelineContext) -> VisualizationResult:
        """Generate annotated images from a completed pipeline context.

        Args:
            context: Completed pipeline context containing all results.

        Returns:
            VisualizationResult with the composite image and named layers.
        """
        ...


class IExportStrategy(ABC):
    """Contract for result export strategies (CSV, PDF, DICOM SR, etc.)."""

    @abstractmethod
    def export(self, context: PipelineContext, destination: Any) -> None:
        """Persist the pipeline results to ``destination``.

        Args:
            context: Completed pipeline context.
            destination: File path, stream or connection depending on impl.
        """
        ...
