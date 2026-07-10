"""CervixAI V2 — Segmentation pipeline stage.

Wraps the ISegmenter strategy so it integrates into the PipelineStage
system.  The segmenter itself is injected via DI.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from core.context import PipelineContext
from core.interfaces import ISegmenter
from core.pipeline_manager import PipelineStage

logger = logging.getLogger(__name__)


class SegmentationStage(PipelineStage):
    """Pipeline stage that delegates inference to an ISegmenter.

    Args:
        segmenter: The injected segmentation backend.
    """

    def __init__(self, segmenter: ISegmenter) -> None:
        """Initialise with injected segmenter."""
        self._segmenter = segmenter

    def execute(self, context: PipelineContext) -> None:
        """Run segmentation and store the result in context.

        Reads ``context.original_image``, runs inference, and stores:
        - ``context.segmentation`` (SegmentationResult)
        - artifact ``"prediction"`` (raw integer mask)

        Args:
            context: The shared pipeline context.
        """
        if context.original_image is None:
            raise ValueError("SegmentationStage: original_image is None")

        result = self._segmenter.predict(context.original_image)
        context.segmentation = result
        context.register("prediction", result.prediction)

        logger.debug(
            "SegmentationStage [%s]: classes=%s (%.3fs)",
            result.model_backend,
            np.unique(result.prediction).tolist(),
            result.execution_time,
        )
