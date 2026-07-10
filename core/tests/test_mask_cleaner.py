"""Tests for MaskCleaningStage."""

import numpy as np
import pytest
from core.config import PostprocessingConfig
from core.context import PipelineContext
from core.domain import SegmentationResult
from core.postprocessing.mask_cleaner import MaskCleaningStage


def _make_segmentation(h=100, w=150):
    pred = np.zeros((h, w), dtype=np.uint8)
    # Anterior lip (class 1) — solid rectangle
    pred[20:40, 20:120] = 1
    # Posterior lip (class 2) — solid rectangle
    pred[60:80, 20:120] = 2
    return SegmentationResult(
        prediction=pred,
        probability_map=None,
        model_backend="test",
        execution_time=0.0,
    )


def test_artifacts_registered():
    ctx = PipelineContext()
    ctx.segmentation = _make_segmentation()
    stage = MaskCleaningStage(PostprocessingConfig())
    stage.execute(ctx)
    assert ctx.has_artifact("mask_anterior")
    assert ctx.has_artifact("mask_posterior")
    assert ctx.has_artifact("clean_mask")


def test_clean_mask_has_correct_classes():
    ctx = PipelineContext()
    ctx.segmentation = _make_segmentation()
    stage = MaskCleaningStage(PostprocessingConfig())
    stage.execute(ctx)
    clean = ctx.get_artifact("clean_mask")
    assert set(np.unique(clean)).issubset({0, 1, 2})


def test_raises_if_no_segmentation():
    ctx = PipelineContext()
    stage = MaskCleaningStage(PostprocessingConfig())
    with pytest.raises(ValueError):
        stage.execute(ctx)


def test_handles_empty_mask():
    pred = np.zeros((100, 150), dtype=np.uint8)
    ctx = PipelineContext()
    ctx.segmentation = SegmentationResult(
        prediction=pred,
        probability_map=None,
        model_backend="test",
        execution_time=0.0,
    )
    stage = MaskCleaningStage(PostprocessingConfig())
    stage.execute(ctx)
    clean = ctx.get_artifact("clean_mask")
    assert clean is not None
    assert clean.sum() == 0
