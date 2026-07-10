"""CervixAI V2 — Measurement engine orchestrator (pipeline stage).

Orchestrates the full anatomical measurement sequence:

    Clean Mask
        ↓
    Canal Gap Extraction
        ↓
    Skeletonization
        ↓
    Skeleton Pruning
        ↓
    Graph Simplification
        ↓
    Longest Geodesic Path (Dijkstra on pruned graph)
        ↓
    B-Spline Fitting
        ↓
    Arc-Length Measurement

This stage reads the cleaned mask from the context artifact registry
and writes a ``MeasurementResult`` domain object back into the context.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from core.config import MeasurementConfig
from core.context import PipelineContext
from core.domain import MeasurementResult
from core.measurement.canal_extractor import CanalCorridorExtractor
from core.measurement.centerline_extractor import CenterlineExtractor
from core.measurement.skeleton_extractor import SkeletonExtractor
from core.measurement.spline_measurement import SplineMeasurement
from core.pipeline_manager import PipelineStage

logger = logging.getLogger(__name__)


class MeasurementStage(PipelineStage):
    """Pipeline stage that orchestrates anatomical canal measurement.

    All sub-components are injected at construction time, enabling
    independent testing of each step.

    Args:
        config: Measurement configuration.
    """

    def __init__(self, config: MeasurementConfig) -> None:
        """Initialise with injected sub-components."""
        self._cfg = config
        self._corridor_extractor = CanalCorridorExtractor(config)
        self._skeleton_extractor = SkeletonExtractor(config.min_branch_length)
        self._centerline_extractor = CenterlineExtractor()
        self._spline = SplineMeasurement()

    def execute(self, context: PipelineContext) -> None:
        """Run the full measurement sequence.

        Reads ``"clean_mask"`` artifact from the context registry.
        Writes:
        - ``context.measurement`` (MeasurementResult)
        - artifacts: ``"canal_gap"``, ``"skeleton"``, ``"centerline_pts"``

        Args:
            context: The shared pipeline context.
        """
        t0 = time.perf_counter()

        clean_mask = context.get_artifact("clean_mask")
        if clean_mask is None:
            raise ValueError("MeasurementStage: 'clean_mask' artifact not found")

        result = self._run_pipeline(clean_mask, context)
        if result is None:
            context.add_warning(
                "MeasurementStage: could not compute measurement "
                "(gap, skeleton or spline extraction failed)"
            )
            return

        result = self._attach_mm_conversion(result, context)
        context.measurement = result
        elapsed = time.perf_counter() - t0
        logger.info(
            "MeasurementStage: arc_length=%.1f px, arc_length_mm=%s (%.3fs)",
            result.arc_length_px,
            f"{result.arc_length_mm:.2f}" if result.arc_length_mm else "N/A",
            elapsed,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_pipeline(
        self, clean_mask: np.ndarray, context: PipelineContext
    ) -> Optional[MeasurementResult]:
        """Execute the measurement sub-pipeline.

        Args:
            clean_mask: Integer mask {0, 1, 2}.
            context: Pipeline context for artifact registration.

        Returns:
            MeasurementResult or None on failure.
        """
        corridor = self._corridor_extractor.extract(clean_mask)
        if corridor is None:
            return None
        context.register("canal_gap", corridor)

        skeleton = self._skeleton_extractor.extract(corridor)
        if skeleton is None:
            return None
        context.register("skeleton", skeleton.astype(np.uint8) * 255)

        centerline = self._centerline_extractor.extract(skeleton)
        if centerline is None or len(centerline) < 2:
            return None
        context.register("centerline_pts", centerline)

        spline_points, arc_length = self._spline.fit_spline(centerline)
        context.register("spline_points", spline_points)

        # Endpoints of the axis are OI/OE; centerline points are (y, x).
        # Convention: OI on the left (smaller x), OE on the right.
        oi_yx, oe_yx = centerline[0], centerline[-1]
        if oi_yx[1] > oe_yx[1]:
            oi_yx, oe_yx = oe_yx, oi_yx
        oi_xy = (int(oi_yx[1]), int(oi_yx[0]))
        oe_xy = (int(oe_yx[1]), int(oe_yx[0]))

        return MeasurementResult(
            internal_os=oi_xy,
            external_os=oe_xy,
            canal_axis=np.asarray(centerline),
            spline_points=np.asarray(spline_points),
            arc_length_px=float(arc_length),
            arc_length_mm=None,  # filled in by _attach_mm_conversion
            geodesic_path_indices=[],
            execution_time=0.0,
        )

    def _attach_mm_conversion(
        self, result: MeasurementResult, context: PipelineContext
    ) -> MeasurementResult:
        """Attach mm conversion if calibration is available.

        Args:
            result: Preliminary MeasurementResult with arc_length_px set.
            context: Context containing calibration result.

        Returns:
            Updated MeasurementResult with arc_length_mm filled if possible.
        """
        if (
            context.calibration is not None
            and context.calibration.is_available
            and context.calibration.mm_per_pixel is not None
        ):
            result.arc_length_mm = (
                result.arc_length_px * context.calibration.mm_per_pixel
            )
        return result
