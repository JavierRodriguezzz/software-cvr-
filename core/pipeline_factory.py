from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from core.calibration.calibration_engine import CalibrationStage
from core.config import AppConfig, get_config
from core.confidence.confidence_engine import ConfidenceStage
from core.context import PipelineContext
from core.debug.artifact_saver import save_debug_artifacts
from core.measurement.measurement_engine import MeasurementStage
from core.pipeline_manager import PipelineManager
from core.postprocessing.mask_cleaner import MaskCleaningStage
from core.segmentation.segmentation_stage import SegmentationStage
from core.segmentation.unet_segmenter import UNetSegmenter
from core.segmentation.yolo_segmenter import YOLOSegmenter
from core.visualization.visualization_engine import VisualizationStage

logger = logging.getLogger(__name__)

_segmenter_singleton = None
_config_singleton: Optional[AppConfig] = None


def get_segmenter(config: AppConfig, base_dir: Path):
    global _segmenter_singleton, _config_singleton
    if _segmenter_singleton is None:
        backend = config.segmentation.model_backend
        if backend == "unet":
            weights = base_dir / config.segmentation.unet_weights
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _segmenter_singleton = UNetSegmenter(weights, device)
            logger.info("Segmenter singleton created: UNetSegmenter on %s", device)
        elif backend == "yolo":
            weights = str(base_dir / config.segmentation.yolo_weights)
            _segmenter_singleton = YOLOSegmenter(weights, config.segmentation.yolo_imgsz, config.segmentation.yolo_conf)
            logger.info("Segmenter singleton created: YOLOSegmenter (stub)")
        else:
            raise ValueError(f"Unknown model_backend: '{backend}'")
        _config_singleton = config
    return _segmenter_singleton


def build_pipeline(base_dir: Path, config: Optional[AppConfig] = None) -> PipelineManager:
    cfg = config or get_config()
    segmenter = get_segmenter(cfg, base_dir)
    return PipelineManager(stages=[
        SegmentationStage(segmenter),
        MaskCleaningStage(cfg.postprocessing),
        # Calibration must run BEFORE measurement so the mm conversion,
        # which MeasurementStage applies, sees the scale factor.
        CalibrationStage(cfg.calibration),
        MeasurementStage(cfg.measurement),
        ConfidenceStage(cfg.confidence),
        VisualizationStage(),
    ])


def run_pipeline(image_bgr, base_dir: Path, metadata: dict) -> PipelineContext:
    pipeline = build_pipeline(base_dir)
    context = PipelineContext(original_image=image_bgr, metadata=metadata)
    pipeline.run(context)

    _maybe_save_debug(context, base_dir, metadata)
    return context


def _maybe_save_debug(context: PipelineContext, base_dir: Path, metadata: dict) -> None:
    """Save debug artifacts if debug mode is enabled in config."""
    try:
        cfg = get_config()
        if not cfg.debug.enabled:
            return
        run_id = _debug_run_id(metadata)
        dst = base_dir / cfg.debug.output_dir / run_id
        save_debug_artifacts(context, run_id, dst, cfg.debug)
    except Exception as exc:
        logger.warning("Debug artifact saving failed (non-fatal): %s", exc)


def _debug_run_id(metadata: dict) -> str:
    """Build a unique run ID from metadata."""
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_id = metadata.get("file_id", metadata.get("filename", "run"))
    stem = Path(str(file_id)).stem
    return f"{stem}_{ts}"
