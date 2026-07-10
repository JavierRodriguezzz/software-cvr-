"""CervixAI V2 — Strongly-typed domain objects.

Every result produced by a pipeline stage is represented as a
dataclass, eliminating raw dict passing between components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

@dataclass
class SegmentationResult:
    """Output of the segmentation stage.

    Attributes:
        prediction: Integer mask (H, W) with class values {0=background,
            1=anterior lip, 2=posterior lip}.
        probability_map: Optional softmax probability map (C, H, W) or None
            when the backend does not expose per-class probabilities.
        model_backend: Name of the model used (e.g. ``"unet"``, ``"yolo"``).
        execution_time: Wall-clock seconds taken by this stage.
    """

    prediction: np.ndarray
    probability_map: Optional[np.ndarray]
    model_backend: str
    execution_time: float


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

@dataclass
class MeasurementResult:
    """Output of the anatomical measurement stage.

    The canal is NOT modelled as a straight segment from OI to OE.
    The ``canal_axis`` field holds the ordered sequence of pixel coordinates
    that constitute the anatomical axis (the central line of the canal gap),
    which may be straight or curved depending on the patient.

    Attributes:
        internal_os: (x, y) pixel coordinate of the Internal Os (OI).
            Determined as the endpoint of the longest geodesic path on the
            canal skeleton, NOT as the leftmost/rightmost contour point.
        external_os: (x, y) pixel coordinate of the External Os (OE).
        canal_axis: Ordered array of (x, y) skeleton pixels forming the
            anatomical axis of the canal, from OI to OE.
        spline_points: Dense (x, y) points sampled from the fitted B-Spline
            for rendering. Follows the curve of the canal.
        arc_length_px: Arc-length of the B-Spline in pixels (the official
            cervical length measurement).
        arc_length_mm: Arc-length converted to millimetres using the
            calibration scale. ``None`` if calibration is unavailable.
        geodesic_path_indices: Node indices of the Dijkstra geodesic path
            on the pruned skeleton graph (for debugging / verification).
        execution_time: Wall-clock seconds taken by this stage.
    """

    internal_os: Tuple[int, int]
    external_os: Tuple[int, int]
    canal_axis: np.ndarray
    spline_points: np.ndarray
    arc_length_px: float
    arc_length_mm: Optional[float]
    geodesic_path_indices: List[int]
    execution_time: float


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """Output of the calibration stage.

    Attributes:
        mm_per_pixel: Conversion factor. ``None`` if the scale bar could not
            be detected.
        scale_bar_bbox: (x, y, w, h) bounding box of the detected scale bar
            in the original image coordinate system. ``None`` on failure.
        method: Description of the detection method used.
        execution_time: Wall-clock seconds taken by this stage.
    """

    mm_per_pixel: Optional[float]
    scale_bar_bbox: Optional[Tuple[int, int, int, int]]
    method: str
    execution_time: float

    @property
    def is_available(self) -> bool:
        """Return True when a valid scale factor was detected."""
        return self.mm_per_pixel is not None


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceResult:
    """Output of the confidence engine.

    Attributes:
        score: Overall confidence score in the range [0, 100].
        components: Per-factor scores contributing to the overall score.
            Keys: ``mean_probability``, ``skeleton_continuity``,
            ``anatomical_plausibility``, ``geometric_consistency``,
            ``calibration_bonus``.
        execution_time: Wall-clock seconds taken by this stage.
    """

    score: float
    components: Dict[str, float]
    execution_time: float

    @property
    def label(self) -> str:
        """Human-readable confidence label."""
        if self.score >= 80:
            return "Alta"
        if self.score >= 55:
            return "Media"
        return "Baja"


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

@dataclass
class VisualizationResult:
    """Output of the visualization engine.

    Attributes:
        composite: Full annotated composite image (H, W, 3) BGR.
        layers: Individual overlay layers keyed by name so that the
            pipeline can persist each one independently.
            Expected keys: ``"clean_mask"``, ``"skeleton"``,
            ``"centerline"``, ``"spline_overlay"``.
        execution_time: Wall-clock seconds taken by this stage.
    """

    composite: np.ndarray
    layers: Dict[str, np.ndarray] = field(default_factory=dict)
    execution_time: float = 0.0


# ---------------------------------------------------------------------------
# Pipeline Report
# ---------------------------------------------------------------------------

@dataclass
class PipelineReport:
    """Per-stage timing report in milliseconds.

    Usage::

        report = PipelineReport.from_stage_times(context._stage_times)
        print(report.preprocessing_stage)   # 13.2
        print(report.stage_times_ms)        # {"PreprocessingStage": 13.2, ...}

    Each attribute is dynamically accessible by stage class name
    (converted to snake_case).  The raw dict is always available via
    ``stage_times_ms``.
    """

    stage_times_ms: Dict[str, float] = field(default_factory=dict)
    _total_seconds: float = 0.0

    @classmethod
    def from_stage_times(
        cls, stage_times: Dict[str, float], total_seconds: float = 0.0
    ) -> "PipelineReport":
        """Build a report from the raw seconds-per-stage dict.

        Args:
            stage_times: Dict mapping stage class names to elapsed seconds.
            total_seconds: Total pipeline wall-clock time in seconds.

        Returns:
            A PipelineReport with values converted to milliseconds (1 d.p.).
        """
        return cls(
            stage_times_ms={
                name: round(sec * 1000.0, 1) for name, sec in stage_times.items()
            },
            _total_seconds=total_seconds,
        )

    @property
    def total_ms(self) -> float:
        """Sum of all recorded stage times in milliseconds."""
        return round(sum(self.stage_times_ms.values()), 1)

    @property
    def total_s(self) -> float:
        """Total pipeline wall-clock time in seconds."""
        return round(self._total_seconds, 3)

    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access by class name or snake_case.

        ``report.PreprocessingStage`` and ``report.preprocessing_stage``
        both map to ``report.stage_times_ms["PreprocessingStage"]``.
        """
        if name in self.stage_times_ms:
            return self.stage_times_ms[name]
        cls_name = "".join(p.capitalize() for p in name.split("_"))
        if cls_name in self.stage_times_ms:
            return self.stage_times_ms[cls_name]
        raise AttributeError(
            f"PipelineReport has no stage '{name}' or '{cls_name}'"
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable dict."""
        return {
            "stage_times_ms": self.stage_times_ms,
            "total_ms": self.total_ms,
            "total_s": self.total_s,
        }
