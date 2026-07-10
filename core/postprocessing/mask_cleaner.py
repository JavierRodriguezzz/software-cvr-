from __future__ import annotations

import logging

import cv2
import numpy as np
from scipy import ndimage as ndi

from core.config import PostprocessingConfig
from core.context import PipelineContext
from core.pipeline_manager import PipelineStage

logger = logging.getLogger(__name__)


class MaskCleaningStage(PipelineStage):
    def __init__(self, config: PostprocessingConfig) -> None:
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.morph_kernel_size, config.morph_kernel_size))
        self._min_area = config.min_component_area
        self._sigma = config.boundary_smooth_sigma

    def execute(self, context: PipelineContext) -> None:
        if context.segmentation is None:
            raise ValueError("MaskCleaningStage: no segmentation result in context")
        pred = context.segmentation.prediction
        mask_ant = self._clean_class(pred, 1)
        mask_post = self._clean_class(pred, 2)
        context.register("mask_anterior", mask_ant)
        context.register("mask_posterior", mask_post)
        clean = np.zeros_like(pred)
        clean[mask_ant > 0] = 1
        clean[mask_post > 0] = 2
        context.register("clean_mask", clean)
        context.add_log("MaskCleaningStage completed")

    def _clean_class(self, pred: np.ndarray, class_val: int) -> np.ndarray:
        mask = (pred == class_val).astype(np.uint8) * 255
        if mask.sum() == 0:
            return mask
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, self._kernel)
        filled = cv2.morphologyEx(opened, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        blurred = cv2.GaussianBlur(filled.astype(np.float32), (0, 0), self._sigma)
        smoothed = (blurred > 127).astype(np.uint8) * 255
        labeled, n = ndi.label(smoothed)
        if n == 0:
            return np.zeros_like(mask)
        sizes = ndi.sum(smoothed, labeled, range(1, n + 1))
        largest = int(np.argmax(sizes)) + 1
        return (labeled == largest).astype(np.uint8) * 255
