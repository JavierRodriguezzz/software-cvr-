"""CervixAI V2 — Canal axis (equidistant contact band) extraction.

The endocervical canal is NOT a segmented tissue class, and it does NOT
reliably appear as a dark gap: the anterior lip (class 1) and the
posterior lip (class 2) usually TOUCH along the canal, which shows up as
a thin echogenic interface rather than empty space (see O-CCR, Hwangbo
2025).  Requiring background pixels between the lips therefore collapses
the canal to a couple of pixels.

Instead, the canal axis is reconstructed geometrically as the locus of
points EQUIDISTANT from the two lips (the medial line / SKIZ between
them), restricted to the zone where both lips are near.  This works
whether the lips touch (the axis follows the contact line) or leave a
small gap (the axis runs down the middle of it).  Skeletonising the
resulting band yields the 1-px axis from the internal os (OI) to the
external os (OE).
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage as ndi

from core.config import MeasurementConfig

logger = logging.getLogger(__name__)

_CLASS_ANTERIOR = 1
_CLASS_POSTERIOR = 2
_CONN8 = np.ones((3, 3), dtype=int)


class CanalCorridorExtractor:
    """Extract the canal axis band from a cleaned segmentation mask.

    The approach:
    1. Separate anterior (class 1) and posterior (class 2) binary masks.
    2. Compute the distance transform to each lip.
    3. The canal axis is where those two distances are (nearly) equal —
       the medial line between the lips — restricted to the ``zone`` where
       both lips are near (so the line does not extend to the image edges).
    4. Keep only the largest connected component (8-connectivity, since
       the axis is often diagonal and thin).

    Args:
        config: Measurement configuration.
    """

    def __init__(self, config: MeasurementConfig) -> None:
        """Initialise with injected configuration."""
        self._cfg = config

    def extract(self, clean_mask: np.ndarray) -> Optional[np.ndarray]:
        """Extract the binary canal axis band.

        Args:
            clean_mask: Integer mask (H, W) with values {0, 1, 2}.

        Returns:
            Binary uint8 mask of the canal axis band (values 0/255), or
            ``None`` if it could not be formed (a lip missing, or the lips
            never come near within the zone).
        """
        ant = clean_mask == _CLASS_ANTERIOR
        post = clean_mask == _CLASS_POSTERIOR

        if not ant.any() or not post.any():
            logger.warning("CanalCorridorExtractor: one or both lips missing")
            return None

        band = self._contact_band(ant, post)
        if band is None:
            return None

        largest = self._largest_component(band)
        if largest.sum() == 0:
            logger.warning("CanalCorridorExtractor: contact band empty after LCC")
            return None

        logger.debug(
            "CanalCorridorExtractor: axis band extracted (%d px)",
            int(np.count_nonzero(largest)),
        )
        return (largest * 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _contact_band(self, ant: np.ndarray, post: np.ndarray) -> Optional[np.ndarray]:
        """Build the equidistant band between the two lips.

        Args:
            ant: Boolean anterior-lip mask.
            post: Boolean posterior-lip mask.

        Returns:
            Binary uint8 band, or ``None`` if the lips never overlap within
            the configured zone dilation.
        """
        dist_ant = ndi.distance_transform_edt(~ant)
        dist_post = ndi.distance_transform_edt(~post)

        k = self._cfg.gap_dilation_kernel
        kernel = np.ones((k, k), np.uint8)
        iters = self._cfg.contact_zone_iters
        zone = (
            cv2.dilate(ant.astype(np.uint8), kernel, iterations=iters)
            & cv2.dilate(post.astype(np.uint8), kernel, iterations=iters)
        ) > 0
        if not zone.any():
            logger.warning(
                "CanalCorridorExtractor: lips never overlap within %d dilations", iters
            )
            return None

        equidistant = np.abs(dist_ant - dist_post) <= self._cfg.contact_band_px
        band = zone & equidistant
        return band.astype(np.uint8)

    def _largest_component(self, binary: np.ndarray) -> np.ndarray:
        """Keep only the largest 8-connected component.

        Args:
            binary: Binary uint8 mask (values 0/1 or 0/255).

        Returns:
            Binary uint8 mask (values 0/1) with only the largest CC.
        """
        labeled, n = ndi.label(binary > 0, structure=_CONN8)
        if n == 0:
            return np.zeros_like(binary)
        sizes = ndi.sum(binary > 0, labeled, range(1, n + 1))
        largest_lbl = int(np.argmax(sizes)) + 1
        return (labeled == largest_lbl).astype(np.uint8)
