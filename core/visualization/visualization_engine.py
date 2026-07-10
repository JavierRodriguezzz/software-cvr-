from __future__ import annotations

import logging

import cv2
import numpy as np

from core.context import PipelineContext
from core.domain import VisualizationResult
from core.pipeline_manager import PipelineStage

logger = logging.getLogger(__name__)


class VisualizationStage(PipelineStage):
    def execute(self, context: PipelineContext) -> None:
        original = context.original_image
        if original is None:
            return
        vis = original.copy()
        overlay = np.zeros_like(vis, dtype=np.uint8)
        ant_mask = context.get_artifact("mask_anterior")
        post_mask = context.get_artifact("mask_posterior")
        if ant_mask is not None:
            overlay[ant_mask > 0] = (0, 0, 255)
        if post_mask is not None:
            overlay[post_mask > 0] = (0, 255, 0)
        vis = cv2.addWeighted(vis, 1.0, overlay, 0.35, 0)
        skel = context.get_artifact("skeleton")
        if skel is not None and skel.sum() > 0:
            vis[skel > 0] = (255, 255, 0)
        cpts = context.get_artifact("centerline_pts")
        if cpts is not None and len(cpts) > 1:
            pts = cpts[:, ::-1].astype(np.int32)
            cv2.polylines(vis, [pts], False, (0, 255, 255), 2)
        spline = context.get_artifact("spline_points")
        if spline is not None and len(spline) > 1:
            spts = spline[:, ::-1].astype(np.int32)
            cv2.polylines(vis, [spts], False, (255, 0, 255), 2)
        meas = context.measurement
        if meas is not None:
            oi = meas.internal_os
            oe = meas.external_os
            cv2.circle(vis, oi, 6, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(vis, "OI", (oi[0] - 25, oi[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(vis, oe, 6, (255, 0, 0), -1, cv2.LINE_AA)
            cv2.putText(vis, "OE", (oe[0] + 10, oe[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cal = context.calibration
            if cal and cal.is_available and cal.mm_per_pixel:
                cm = meas.arc_length_px * cal.mm_per_pixel / 10.0
                txt = f"Length: {cm:.2f} cm ({meas.arc_length_px:.1f} px)"
            else:
                txt = f"Length: {meas.arc_length_px:.1f} px"
            cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        conf = context.confidence
        if conf is not None:
            cv2.putText(vis, f"Confidence: {conf.score:.1f}% ({conf.label})",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        clean_mask = context.get_artifact("clean_mask")
        context.visualization = VisualizationResult(
            composite=vis,
            layers={
                "clean_mask": (clean_mask * 127).astype(np.uint8) if clean_mask is not None else np.zeros_like(original),
                "skeleton": (skel * 255).astype(np.uint8) if skel is not None else np.zeros((*original.shape[:2],), dtype=np.uint8),
                "centerline_overlay": cpts if cpts is not None else np.zeros((0, 2)),
                "spline_overlay": spline if spline is not None else np.zeros((0, 2)),
            },
            execution_time=0.0,
        )
        context.add_log("VisualizationStage completed")
