"""CervixAI V2 — Debug artifact saver.

When ``config.debug.enabled`` is True, ``save_debug_artifacts``
writes every intermediate image and JSON log into a timestamped
subdirectory under the configured ``output_dir``, one subdirectory
per pipeline run.

This is a **pure side-effect** function — it has no pipeline stage,
it never reads or modifies the context, and it can be disabled
entirely from ``config.yaml`` without touching any stage code.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from core.config import DebugConfig
from core.context import PipelineContext
from core.domain import PipelineReport

logger = logging.getLogger(__name__)


def save_debug_artifacts(
    context: PipelineContext,
    run_id: str,
    dst: Path,
    cfg: DebugConfig,
) -> None:
    """Write all intermediate pipeline outputs to ``dst``.

    Args:
        context: Fully populated pipeline context.
        run_id: Unique identifier for this run (e.g. file stem + timestamp).
        dst: Root output directory (from ``config.debug.output_dir``).
        cfg: Debug configuration (controls which files are saved).
    """
    dst.mkdir(parents=True, exist_ok=True)

    if cfg.save_images:
        _save_images(context, dst, run_id)

    if cfg.save_json:
        _save_jsons(context, dst, run_id, cfg)

    logger.info("Debug artifacts saved to %s (%d files)", dst, len(list(dst.iterdir())))


# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------

def _save_images(ctx: PipelineContext, dst: Path, run_id: str) -> None:
    images: Dict[str, Optional[np.ndarray]] = {
        "original": ctx.original_image,
        "preprocessed": ctx.preprocessed_image,
        "prediction": _prediction_rgb(ctx),
        "clean_mask": _clean_mask_rgb(ctx),
        "skeleton": ctx.get_artifact("skeleton"),
        "centerline": _centerline_rgb(ctx),
        "spline": _spline_rgb(ctx),
    }

    if ctx.visualization is not None:
        images["measurement"] = ctx.visualization.composite

    for name, arr in images.items():
        if arr is None:
            continue
        path = dst / f"{run_id}_{name}.png"
        success = cv2.imwrite(str(path), arr)
        if not success:
            logger.warning("Failed to write debug image: %s", path)


def _prediction_rgb(ctx: PipelineContext) -> Optional[np.ndarray]:
    if ctx.segmentation is None:
        return None
    pred = ctx.segmentation.prediction
    if pred.ndim == 2:
        return (pred / 2 * 255).astype(np.uint8) if pred.max() > 0 else None
    return None


def _clean_mask_rgb(ctx: PipelineContext) -> Optional[np.ndarray]:
    clean = ctx.get_artifact("clean_mask")
    if clean is None or clean.max() == 0:
        return None
    return (clean / 2 * 255).astype(np.uint8)


def _centerline_rgb(ctx: PipelineContext) -> Optional[np.ndarray]:
    cpts = ctx.get_artifact("centerline_pts")
    if cpts is None or len(cpts) < 2:
        return None
    h, w = ctx.original_image.shape[:2] if ctx.original_image is not None else (336, 544)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    pts = cpts[:, ::-1].astype(np.int32)
    cv2.polylines(canvas, [pts], False, (0, 255, 255), 2)
    return canvas


def _spline_rgb(ctx: PipelineContext) -> Optional[np.ndarray]:
    if ctx.measurement is None:
        return None
    spline = ctx.measurement.spline_points
    if spline is None or len(spline) < 2:
        return None
    h, w = ctx.original_image.shape[:2] if ctx.original_image is not None else (336, 544)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    pts = spline[:, ::-1].astype(np.int32)
    cv2.polylines(canvas, [pts], False, (255, 0, 255), 2)
    return canvas


# ---------------------------------------------------------------------------
# JSON saving
# ---------------------------------------------------------------------------

def _save_jsons(ctx: PipelineContext, dst: Path, run_id: str, cfg: DebugConfig) -> None:
    summary = ctx.to_summary_dict()
    report = ctx.pipeline_report

    summary["pipeline_report"] = report.to_dict()

    # confidence.json
    if ctx.confidence is not None:
        conf_path = dst / f"{run_id}_confidence.json"
        _write_json(conf_path, {
            "score": round(ctx.confidence.score, 1),
            "label": ctx.confidence.label,
            "components": {
                k: round(v, 3) for k, v in ctx.confidence.components.items()
            },
        })

    # execution_log.json — full summary
    log_path = dst / f"{run_id}_execution_log.json"
    _write_json(log_path, summary)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
