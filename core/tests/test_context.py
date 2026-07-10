"""Tests for PipelineContext artifact registry."""

import pytest
import numpy as np
from core.context import PipelineContext


def test_register_and_retrieve_artifact():
    ctx = PipelineContext()
    img = np.zeros((10, 10), dtype=np.uint8)
    ctx.register("test_img", img)
    retrieved = ctx.get_artifact("test_img")
    assert retrieved is img


def test_has_artifact_true():
    ctx = PipelineContext()
    ctx.register("key", 42)
    assert ctx.has_artifact("key") is True


def test_has_artifact_false():
    ctx = PipelineContext()
    assert ctx.has_artifact("nonexistent") is False


def test_get_artifact_default():
    ctx = PipelineContext()
    result = ctx.get_artifact("missing", default="fallback")
    assert result == "fallback"


def test_add_warning_and_error():
    ctx = PipelineContext()
    ctx.add_warning("test warning")
    ctx.add_error("test error")
    assert len(ctx.warnings) == 1
    assert len(ctx.errors) == 1
    assert ctx.has_errors is True


def test_record_stage_time():
    ctx = PipelineContext()
    ctx.record_stage_time("StageA", 0.123)
    assert "StageA" in ctx.stage_summary
    assert ctx.stage_summary["StageA"] == pytest.approx(0.123)


def test_to_summary_dict_without_results():
    ctx = PipelineContext()
    summary = ctx.to_summary_dict()
    assert "execution_time" in summary
    assert "errors" in summary
    assert "measurement" not in summary


def test_to_summary_dict_with_measurement():
    from core.domain import MeasurementResult
    import numpy as np

    ctx = PipelineContext()
    ctx.measurement = MeasurementResult(
        internal_os=(10, 20),
        external_os=(100, 20),
        canal_axis=np.array([[20, 10], [20, 100]]),
        spline_points=np.zeros((5, 2)),
        arc_length_px=90.0,
        arc_length_mm=None,
        geodesic_path_indices=[],
        execution_time=0.1,
    )
    summary = ctx.to_summary_dict()
    assert summary["measurement"]["arc_length_px"] == pytest.approx(90.0)
    assert summary["measurement"]["oi"] == [10, 20]
