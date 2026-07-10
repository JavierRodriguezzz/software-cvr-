import logging

import numpy as np
from scipy.interpolate import splprep, splev

logger = logging.getLogger(__name__)


class SplineMeasurement:
    def __init__(self, smooth_per_point: float = 1.0, k: int = 3):
        # ``smooth_per_point`` is the spline smoothing budget as allowed
        # squared residual per point (~1 => ~1 px deviation each).  It
        # removes the skeleton's staircase noise without letting the spline
        # oscillate — a too-small value makes the arc length blow up.
        self._smooth_per_point = smooth_per_point
        self._k = k

    def fit_spline(self, points: np.ndarray) -> tuple:
        raw_length = self._arc_length(points)
        if len(points) < 4:
            return points, raw_length
        try:
            m = len(points)
            s = max(m * self._smooth_per_point, 1e-3)
            tck, _ = splprep([points[:, 1], points[:, 0]], s=s, k=min(self._k, m - 1))
            u_fine = np.linspace(0, 1, max(100, m * 2))
            x_fine, y_fine = splev(u_fine, tck)
            spline = np.column_stack([y_fine, x_fine])
            length = self._arc_length(spline)
            # Guard: fall back to the raw polyline length if the fit is
            # degenerate or implausible (spline oscillation / blow-up).
            if not np.isfinite(length) or not (0.5 * raw_length <= length <= 1.3 * raw_length):
                return points, raw_length
            return spline, length
        except Exception:
            return points, raw_length

    @staticmethod
    def _arc_length(points: np.ndarray) -> float:
        if len(points) < 2:
            return 0.0
        diffs = np.diff(points.astype(np.float64), axis=0)
        segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        return float(segment_lengths.sum())

    @staticmethod
    def find_endpoints(centerline: np.ndarray) -> tuple:
        if len(centerline) < 2:
            return (0, 0), (0, 0)
        diff_x = centerline[-1, 1] - centerline[0, 1]
        if diff_x > 0:
            return tuple(centerline[0][::-1]), tuple(centerline[-1][::-1])
        return tuple(centerline[-1][::-1]), tuple(centerline[0][::-1])
