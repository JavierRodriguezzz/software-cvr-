"""CervixAI V2 — PipelineContext as an artifact registry.

PipelineContext is the single mutable object that flows through the
entire pipeline.  It acts as both a typed field container AND an
artifact registry so that stages can store and retrieve named artifacts
without coupling to each other directly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.domain import (
    CalibrationResult,
    ConfidenceResult,
    MeasurementResult,
    PipelineReport,
    SegmentationResult,
    VisualizationResult,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Artifact registry and typed field container for the pipeline.

    Each stage receives the same context instance and is expected to
    read its inputs from it and write its outputs back into it.
    Stages must NEVER return multiple values — all results are stored
    here.

    Typed fields hold the domain objects produced by each major stage.
    The ``_registry`` dict stores arbitrary binary artifacts (images,
    arrays) so that stages can persist intermediate results for
    debugging and visualization without polluting the typed namespace.
    """

    # ----- Raw inputs -----
    original_image: Optional[np.ndarray] = None
    """BGR image as loaded from disk.  Never modified by any stage."""

    gray_image: Optional[np.ndarray] = None
    """Grayscale version of the original image."""

    preprocessed_image: Optional[np.ndarray] = None
    """Preprocessed image ready for inference (CLAHE, denoise, normalize)."""

    # ----- Stage results -----
    segmentation: Optional[SegmentationResult] = None
    measurement: Optional[MeasurementResult] = None
    calibration: Optional[CalibrationResult] = None
    confidence: Optional[ConfidenceResult] = None
    visualization: Optional[VisualizationResult] = None

    # ----- Pipeline metadata -----
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Arbitrary key-value metadata from the caller (file_id, patient, etc.)."""

    logs: List[str] = field(default_factory=list)
    """Ordered list of log messages appended by each stage."""

    warnings: List[str] = field(default_factory=list)
    """Non-fatal warnings raised during processing."""

    errors: List[str] = field(default_factory=list)
    """Fatal errors that caused a stage to abort."""

    execution_time: float = 0.0
    """Total wall-clock seconds for the complete pipeline run."""

    # ----- Internal registry -----
    _registry: Dict[str, Any] = field(default_factory=dict, repr=False)
    _stage_times: Dict[str, float] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Artifact registry interface
    # ------------------------------------------------------------------

    def register(self, key: str, artifact: Any) -> None:
        """Store an artifact under ``key``.

        Args:
            key: Unique identifier for this artifact.
            artifact: Any object to store (typically an np.ndarray).
        """
        self._registry[key] = artifact
        logger.debug("Artifact registered: %s", key)

    def get_artifact(self, key: str, default: Any = None) -> Any:
        """Retrieve a previously registered artifact.

        Args:
            key: The artifact identifier.
            default: Value returned when the key is not found.

        Returns:
            The stored artifact or ``default``.
        """
        return self._registry.get(key, default)

    def has_artifact(self, key: str) -> bool:
        """Return True if an artifact with the given key exists."""
        return key in self._registry

    def record_stage_time(self, stage_name: str, elapsed: float) -> None:
        """Record wall-clock time for an individual stage.

        Args:
            stage_name: Human-readable stage identifier.
            elapsed: Seconds elapsed during that stage.
        """
        self._stage_times[stage_name] = elapsed
        self.logs.append(f"[{stage_name}] completed in {elapsed:.3f}s")

    def add_log(self, message: str) -> None:
        """Append an informational log message."""
        self.logs.append(message)
        logger.info(message)

    def add_warning(self, message: str) -> None:
        """Append a non-fatal warning."""
        self.warnings.append(message)
        logger.warning(message)

    def add_error(self, message: str) -> None:
        """Append a fatal error description."""
        self.errors.append(message)
        logger.error(message)

    @property
    def has_errors(self) -> bool:
        """Return True when at least one fatal error has been recorded."""
        return len(self.errors) > 0

    @property
    def stage_summary(self) -> Dict[str, float]:
        """Return a copy of per-stage execution times in seconds."""
        return dict(self._stage_times)

    @property
    def pipeline_report(self) -> PipelineReport:
        """Build and return a PipelineReport from recorded stage times."""
        return PipelineReport.from_stage_times(self._stage_times, self.execution_time)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary of the pipeline run.

        Used by the Flask layer to populate ``meta`` without exposing
        numpy arrays.
        """
        summary: Dict[str, Any] = {
            "execution_time": round(self.execution_time, 3),
            "stage_times": {k: round(v, 3) for k, v in self._stage_times.items()},
            "warnings": self.warnings,
            "errors": self.errors,
        }

        if self.measurement:
            oi = self.measurement.internal_os
            oe = self.measurement.external_os
            summary["measurement"] = {
                "oi": [int(oi[0]), int(oi[1])],
                "oe": [int(oe[0]), int(oe[1])],
                "arc_length_px": round(self.measurement.arc_length_px, 2),
                "arc_length_mm": (
                    round(self.measurement.arc_length_mm, 2)
                    if self.measurement.arc_length_mm is not None
                    else None
                ),
            }

        if self.calibration:
            summary["calibration"] = {
                "mm_per_pixel": self.calibration.mm_per_pixel,
                "method": self.calibration.method,
                "available": self.calibration.is_available,
            }

        if self.confidence:
            summary["confidence"] = {
                "score": round(self.confidence.score, 1),
                "label": self.confidence.label,
                "components": {
                    k: round(v, 3) for k, v in self.confidence.components.items()
                },
            }

        return summary
