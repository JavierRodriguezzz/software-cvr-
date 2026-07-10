"""CervixAI V2 — Centralised configuration loader.

Reads ``config.yaml`` from the project root and exposes a typed
``AppConfig`` dataclass.  If the file is absent, safe defaults are used
so the application can always start.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass
class SegmentationConfig:
    """Segmentation-specific settings."""

    model_backend: str = "unet"
    unet_weights: str = "models/unet_fugc_best.pth"
    unet_input_size: Tuple[int, int] = (544, 336)
    unet_n_channels: int = 1
    unet_n_classes: int = 3
    yolo_weights: str = "models/yolo_cervix_best.pt"
    yolo_imgsz: int = 640
    yolo_conf: float = 0.25


@dataclass
class PreprocessingConfig:
    """Preprocessing pipeline settings."""

    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    denoise_h: int = 7
    denoise_template_window: int = 7
    denoise_search_window: int = 21


@dataclass
class PostprocessingConfig:
    """Mask cleaning settings."""

    min_component_area: int = 200
    morph_kernel_size: int = 9
    boundary_smooth_sigma: float = 1.0


@dataclass
class MeasurementConfig:
    """Canal measurement settings."""

    gap_dilation_kernel: int = 3
    contact_zone_iters: int = 30
    contact_band_px: float = 2.0
    min_branch_length: int = 20
    min_plausible_px: int = 30
    max_plausible_px: int = 2000


@dataclass
class CalibrationConfig:
    """Scale-bar detection settings."""

    known_mm: float = 10.0
    roi_y_fraction: float = 0.70
    roi_x_fraction: float = 0.45
    adaptive_block_size: int = 31
    adaptive_c: int = 10
    min_aspect_ratio: float = 3.0


@dataclass
class ConfidenceConfig:
    """Confidence score weighting."""

    weight_mean_prob: float = 0.25
    weight_skeleton_continuity: float = 0.25
    weight_anatomical_length: float = 0.25
    weight_geometric_consistency: float = 0.15
    weight_calibration_bonus: float = 0.10
    normal_length_min_mm: float = 20.0
    normal_length_max_mm: float = 50.0


@dataclass
class DebugConfig:
    """Debug mode settings for per-stage artifact dumping."""

    enabled: bool = False
    output_dir: str = "debug_output"
    save_images: bool = True
    save_json: bool = True


@dataclass
class AppConfig:
    """Aggregate configuration for the entire CervixAI V2 application."""

    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)


def load_config(path: Optional[Path] = None) -> AppConfig:
    """Load ``AppConfig`` from a YAML file.

    Falls back to all defaults if the file is missing or ``PyYAML`` is
    not installed so the application can always start.

    Args:
        path: Path to the YAML file.  Defaults to ``config.yaml`` in the
            project root.

    Returns:
        Fully populated ``AppConfig`` instance.
    """
    target = path or _CONFIG_PATH

    if not target.exists():
        logger.warning("config.yaml not found at %s — using defaults", target)
        return AppConfig()

    try:
        import yaml  # type: ignore[import]
    except ImportError:
        logger.warning("PyYAML not installed — using default configuration")
        return AppConfig()

    try:
        with target.open("r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}
        return _build_config(raw)
    except Exception as exc:
        logger.error("Failed to parse config.yaml: %s — using defaults", exc)
        return AppConfig()


def _build_config(raw: Dict[str, Any]) -> AppConfig:
    """Construct AppConfig from the parsed YAML dictionary.

    Args:
        raw: Top-level YAML dictionary.

    Returns:
        AppConfig populated from ``raw``.
    """
    seg_raw = raw.get("segmentation", {})
    unet_raw = seg_raw.get("unet", {})
    yolo_raw = seg_raw.get("yolo", {})
    pre_raw = raw.get("preprocessing", {})
    post_raw = raw.get("postprocessing", {})
    meas_raw = raw.get("measurement", {})
    cal_raw = raw.get("calibration", {})
    conf_raw = raw.get("confidence", {})
    debug_raw = raw.get("debug", {})

    unet_size = unet_raw.get("input_size", [544, 336])

    seg = SegmentationConfig(
        model_backend=seg_raw.get("model_backend", "unet"),
        unet_weights=unet_raw.get("weights", "models/unet_fugc_best.pth"),
        unet_input_size=(int(unet_size[0]), int(unet_size[1])),
        unet_n_channels=int(unet_raw.get("n_channels", 1)),
        unet_n_classes=int(unet_raw.get("n_classes", 3)),
        yolo_weights=yolo_raw.get("weights", "models/yolo_cervix_best.pt"),
        yolo_imgsz=int(yolo_raw.get("imgsz", 640)),
        yolo_conf=float(yolo_raw.get("conf", 0.25)),
    )

    tile = pre_raw.get("clahe_tile_grid", [8, 8])
    pre = PreprocessingConfig(
        clahe_clip_limit=float(pre_raw.get("clahe_clip_limit", 2.0)),
        clahe_tile_grid=(int(tile[0]), int(tile[1])),
        denoise_h=int(pre_raw.get("denoise_h", 7)),
        denoise_template_window=int(pre_raw.get("denoise_template_window", 7)),
        denoise_search_window=int(pre_raw.get("denoise_search_window", 21)),
    )

    post = PostprocessingConfig(
        min_component_area=int(post_raw.get("min_component_area", 200)),
        morph_kernel_size=int(post_raw.get("morph_kernel_size", 9)),
        boundary_smooth_sigma=float(post_raw.get("boundary_smooth_sigma", 1.0)),
    )

    meas = MeasurementConfig(
        gap_dilation_kernel=int(meas_raw.get("gap_dilation_kernel", 3)),
        contact_zone_iters=int(meas_raw.get("contact_zone_iters", 30)),
        contact_band_px=float(meas_raw.get("contact_band_px", 2.0)),
        min_branch_length=int(meas_raw.get("min_branch_length", 20)),
        min_plausible_px=int(meas_raw.get("min_plausible_px", 30)),
        max_plausible_px=int(meas_raw.get("max_plausible_px", 2000)),
    )

    cal = CalibrationConfig(
        known_mm=float(cal_raw.get("known_mm", 10.0)),
        roi_y_fraction=float(cal_raw.get("roi_y_fraction", 0.70)),
        roi_x_fraction=float(cal_raw.get("roi_x_fraction", 0.45)),
        adaptive_block_size=int(cal_raw.get("adaptive_block_size", 31)),
        adaptive_c=int(cal_raw.get("adaptive_c", 10)),
        min_aspect_ratio=float(cal_raw.get("min_aspect_ratio", 3.0)),
    )

    conf = ConfidenceConfig(
        weight_mean_prob=float(conf_raw.get("weight_mean_prob", 0.25)),
        weight_skeleton_continuity=float(conf_raw.get("weight_skeleton_continuity", 0.25)),
        weight_anatomical_length=float(conf_raw.get("weight_anatomical_length", 0.25)),
        weight_geometric_consistency=float(conf_raw.get("weight_geometric_consistency", 0.15)),
        weight_calibration_bonus=float(conf_raw.get("weight_calibration_bonus", 0.10)),
        normal_length_min_mm=float(conf_raw.get("normal_length_min_mm", 20.0)),
        normal_length_max_mm=float(conf_raw.get("normal_length_max_mm", 50.0)),
    )

    dbg = DebugConfig(
        enabled=bool(debug_raw.get("enabled", False)),
        output_dir=str(debug_raw.get("output_dir", "debug_output")),
        save_images=bool(debug_raw.get("save_images", True)),
        save_json=bool(debug_raw.get("save_json", True)),
    )

    return AppConfig(
        segmentation=seg,
        preprocessing=pre,
        postprocessing=post,
        measurement=meas,
        calibration=cal,
        confidence=conf,
        debug=dbg,
    )


# Module-level singleton — loaded once at import time.
_default_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Return the module-level config singleton, loading it on first call."""
    global _default_config
    if _default_config is None:
        _default_config = load_config()
    return _default_config
