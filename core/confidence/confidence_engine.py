from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from scipy import ndimage as ndi

from core.config import ConfidenceConfig
from core.context import PipelineContext
from core.domain import ConfidenceResult
from core.pipeline_manager import PipelineStage

logger = logging.getLogger(__name__)


class ConfidenceStage(PipelineStage):
    def __init__(self, config: ConfidenceConfig) -> None:
        self._w_prob = config.weight_mean_prob
        self._w_skel = config.weight_skeleton_continuity
        self._w_anat = config.weight_anatomical_length
        self._w_geom = config.weight_geometric_consistency
        self._w_cal = config.weight_calibration_bonus
        self._norm_min = config.normal_length_min_mm
        self._norm_max = config.normal_length_max_mm

    def execute(self, context: PipelineContext) -> None:
        components: Dict[str, float] = {}
        components["mean_probability"] = self._mean_probability(context)
        components["skeleton_continuity"] = self._skeleton_continuity(context)
        components["anatomical_plausibility"] = self._anatomical_plausibility(context)
        components["geometric_consistency"] = self._geometric_consistency(context)
        components["calibration_bonus"] = self._calibration_bonus(context)
        score = (
            self._w_prob * components["mean_probability"]
            + self._w_skel * components["skeleton_continuity"]
            + self._w_anat * components["anatomical_plausibility"]
            + self._w_geom * components["geometric_consistency"]
            + self._w_cal * components["calibration_bonus"]
        )
        score = min(max(score * 100, 0), 100)
        context.confidence = ConfidenceResult(
            score=round(score, 1),
            components=components,
            execution_time=0.0,
        )
        context.add_log(f"ConfidenceStage: score={score:.1f}%")

    def _mean_probability(self, context: PipelineContext) -> float:
        seg = context.segmentation
        if seg is None or seg.probability_map is None:
            return 0.5
        pmap = seg.probability_map
        if pmap.size == 0 or pmap.max() == 0:
            return 0.5
        return float(np.mean(pmap[pmap > 0])) if np.any(pmap > 0) else 0.5

    def _skeleton_continuity(self, context: PipelineContext) -> float:
        skel_art = context.get_artifact("skeleton")
        if skel_art is None or skel_art.sum() == 0:
            return 0.0
        binary = skel_art > 0
        total = int(binary.sum())
        if total < 2:
            return 0.0
        labeled, n = ndi.label(binary)
        if n == 0:
            return 0.0
        sizes = ndi.sum(binary, labeled, range(1, n + 1))
        largest = int(sizes.max())
        return largest / total

    def _anatomical_plausibility(self, context: PipelineContext) -> float:
        meas = context.measurement
        if meas is None or meas.arc_length_px is None:
            return 0.0
        if context.calibration and context.calibration.is_available and context.calibration.mm_per_pixel:
            length_mm = meas.arc_length_px * context.calibration.mm_per_pixel
        else:
            return 0.5
        if length_mm < self._norm_min:
            return max(0.0, length_mm / self._norm_min)
        if length_mm > self._norm_max:
            return max(0.0, (self._norm_max * 2 - length_mm) / self._norm_max)
        return 1.0

    @staticmethod
    def _geometric_consistency(context: PipelineContext) -> float:
        spline = context.get_artifact("spline_points")
        if spline is None or len(spline) < 4:
            return 0.0
        spline = spline.astype(np.float64)
        diffs = np.diff(spline, axis=0)
        seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
        if len(seg_lens) < 2 or seg_lens.sum() == 0:
            return 0.0
        cv_val = float(np.std(seg_lens)) / float(np.mean(seg_lens))
        return max(0.0, 1.0 - min(cv_val, 1.0))

    @staticmethod
    def _calibration_bonus(context: PipelineContext) -> float:
        if context.calibration is None or not context.calibration.is_available:
            return 0.0
        mmpp = context.calibration.mm_per_pixel
        if mmpp is None:
            return 0.0
        if 0.01 <= mmpp <= 1.0:
            return 1.0
        return 0.5
