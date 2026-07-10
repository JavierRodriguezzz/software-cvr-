from __future__ import annotations

import logging

import cv2
import numpy as np

from core.config import CalibrationConfig
from core.context import PipelineContext
from core.domain import CalibrationResult
from core.pipeline_manager import PipelineStage

logger = logging.getLogger(__name__)


class CalibrationStage(PipelineStage):
    def __init__(self, config: CalibrationConfig) -> None:
        self._known_mm = config.known_mm
        self._roi_y = config.roi_y_fraction
        self._roi_x = config.roi_x_fraction
        self._block = config.adaptive_block_size
        self._c = config.adaptive_c
        self._min_aspect = config.min_aspect_ratio

    def execute(self, context: PipelineContext) -> None:
        # 1. Prefer a physical pixel size supplied by the caller — this is the
        #    DICOM PixelSpacing read at upload, the reliable source of scale.
        supplied = context.metadata.get("mm_per_pixel")
        if supplied is not None and float(supplied) > 0:
            context.calibration = CalibrationResult(
                mm_per_pixel=float(supplied),
                scale_bar_bbox=None,
                method="dicom_pixel_spacing",
                execution_time=0.0,
            )
            context.add_log(f"CalibrationStage: {float(supplied):.4f} mm/px (DICOM)")
            return

        # 2. Fall back to detecting a scale bar in the image (PNG/JPG without DICOM).
        image = context.original_image
        if image is None:
            context.add_warning("CalibrationStage: no image")
            return
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
            h, w = gray.shape
            y_start = int(h * self._roi_y)
            x_end = int(w * self._roi_x)
            roi = gray[y_start:, :x_end]
            if roi.size == 0:
                raise ValueError("Calibration ROI is empty")
            thresh = cv2.adaptiveThreshold(
                roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self._block, self._c
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                raise ValueError("No contours found for calibration")
            candidates = []
            for c in contours:
                x, y, cw, ch = cv2.boundingRect(c)
                if ch == 0:
                    continue
                aspect = cw / float(ch)
                if aspect >= self._min_aspect and cw > 20:
                    candidates.append((cw, (x, y + y_start, cw, ch)))
            if not candidates:
                raise ValueError("No valid scale bar candidate found")
            widest = max(candidates, key=lambda t: t[0])
            bar_width = widest[0]
            bbox = widest[1]
            mm_per_pixel = self._known_mm / float(bar_width)
            context.calibration = CalibrationResult(
                mm_per_pixel=mm_per_pixel,
                scale_bar_bbox=bbox,
                method="adaptive_threshold",
                execution_time=0.0,
            )
            context.add_log(f"CalibrationStage: {mm_per_pixel:.4f} mm/px")
        except ValueError as e:
            context.add_warning(f"Calibration failed: {e}")
            context.calibration = CalibrationResult(
                mm_per_pixel=None, scale_bar_bbox=None, method="failed", execution_time=0.0
            )
