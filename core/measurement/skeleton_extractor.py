from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from skimage.morphology import skeletonize
from scipy import ndimage as ndi

from core.config import MeasurementConfig
from core.context import PipelineContext
from core.pipeline_manager import PipelineStage

logger = logging.getLogger(__name__)


class SkeletonExtractor:
    def __init__(self, min_branch_length: int = 20) -> None:
        self._min_branch_length = min_branch_length

    def extract(self, mask: np.ndarray) -> Optional[np.ndarray]:
        binary = (mask > 0).astype(bool)
        if binary.sum() == 0:
            return None
        skel = skeletonize(binary).astype(np.uint8)
        pruned = self._prune(skel)
        if pruned is None or pruned.sum() == 0:
            return None
        return pruned

    def _prune(self, skel: np.ndarray) -> Optional[np.ndarray]:
        # 8-connectivity: the axis is often a thin diagonal line, which
        # 4-connectivity would fragment into sub-threshold specks.
        labeled, n = ndi.label(skel, structure=np.ones((3, 3), dtype=int))
        if n == 0:
            return None
        sizes = ndi.sum(skel, labeled, range(1, n + 1))
        keep = [i + 1 for i, s in enumerate(sizes) if s >= self._min_branch_length]
        if not keep:
            return None
        return np.isin(labeled, keep).astype(np.uint8)
