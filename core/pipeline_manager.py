"""CervixAI V2 — PipelineManager and PipelineStage ABC.

Each processing step is a PipelineStage that receives and modifies a
PipelineContext.  Stages never return values — all outputs are stored
in the context.  PipelineManager orchestrates the sequence, times each
stage, and handles exceptions in isolation so one failing stage does
not abort the whole pipeline.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional

from core.context import PipelineContext

logger = logging.getLogger(__name__)


class PipelineStage(ABC):
    """Abstract base class for a single processing stage.

    Subclasses must implement :meth:`execute` and should keep it under
    60 lines by delegating complex logic to private helper methods.
    """

    @property
    def name(self) -> str:
        """Return the stage name used in logs and timing records."""
        return self.__class__.__name__

    @abstractmethod
    def execute(self, context: PipelineContext) -> None:
        """Run this stage, reading inputs from and writing outputs to
        ``context``.

        Args:
            context: The shared pipeline context.

        Raises:
            Exception: Any exception is caught by PipelineManager, logged,
                and stored as an error in the context. The pipeline
                continues with the next stage.
        """
        ...


class PipelineManager:
    """Orchestrates an ordered sequence of PipelineStages.

    Usage::

        manager = PipelineManager(stages=[
            PreprocessingStage(config),
            SegmentationStage(segmenter),
            MaskCleaningStage(config),
            MeasurementStage(strategy),
            CalibrationStage(strategy),
            ConfidenceStage(config),
            VisualizationStage(strategy),
        ])
        context = PipelineContext(original_image=img)
        manager.run(context)

    After :meth:`run`, the context contains all results and a timing
    record for each stage.
    """

    def __init__(self, stages: Optional[List[PipelineStage]] = None) -> None:
        """Initialise the manager with an optional list of stages.

        Args:
            stages: Ordered list of stages to execute.  Stages can also
                be appended later via :meth:`add_stage`.
        """
        self._stages: List[PipelineStage] = stages or []

    def add_stage(self, stage: PipelineStage) -> "PipelineManager":
        """Append a stage and return self for fluent chaining.

        Args:
            stage: The stage to append.

        Returns:
            This manager instance.
        """
        self._stages.append(stage)
        return self

    def run(self, context: PipelineContext) -> PipelineContext:
        """Execute all stages in order.

        Each stage is wrapped in a try/except block so that a failure
        in one stage is recorded but does not prevent subsequent stages
        from running.

        Args:
            context: The pipeline context, mutated in-place.

        Returns:
            The same context, fully populated.
        """
        pipeline_start = time.perf_counter()
        logger.info("Pipeline starting (%d stages)", len(self._stages))

        for stage in self._stages:
            self._run_stage(stage, context)

        context.execution_time = time.perf_counter() - pipeline_start
        logger.info(
            "Pipeline finished in %.3fs (errors=%d, warnings=%d)",
            context.execution_time,
            len(context.errors),
            len(context.warnings),
        )
        return context

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_stage(self, stage: PipelineStage, context: PipelineContext) -> None:
        """Run a single stage, capture timing and handle exceptions.

        Args:
            stage: The stage to execute.
            context: The shared context.
        """
        logger.info("Stage starting: %s", stage.name)
        t0 = time.perf_counter()
        try:
            stage.execute(context)
            elapsed = time.perf_counter() - t0
            context.record_stage_time(stage.name, elapsed)
            logger.info("Stage %s OK (%.3fs)", stage.name, elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            context.record_stage_time(stage.name, elapsed)
            msg = f"Stage {stage.name} failed after {elapsed:.3f}s: {exc}"
            context.add_error(msg)
            logger.exception("Stage %s raised an exception", stage.name)
