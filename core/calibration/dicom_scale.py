"""Read the physical pixel size (mm per pixel) from a DICOM dataset.

Ultrasound DICOMs may carry the scale in a few different places.  We try,
in order:

1. ``PixelSpacing`` (mm; the standard tag for calibrated images — confirmed
   present in this project's files, e.g. ``0.087975\\0.087975``).
2. ``SequenceOfUltrasoundRegions[*].PhysicalDeltaX/Y`` (usually cm/pixel;
   converted to mm).  This is the ultrasound-specific calibration.
3. ``ImagerPixelSpacing`` (mm).

Returns a single mm-per-pixel value (the mean of the row/col spacings,
which are equal for the square pixels these scanners produce), or ``None``
if no scale is present.  The value is variable per file, so it must always
be read at runtime — never hard-coded.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# DICOM PhysicalUnits*Direction code 3 == centimetres.
_UNIT_CM = 3


def pixel_spacing_mm_from_dataset(ds) -> Optional[float]:
    """Return mm-per-pixel from a pydicom dataset, or ``None``.

    Args:
        ds: A ``pydicom.dataset.Dataset`` (already read from disk).

    Returns:
        Physical pixel size in millimetres, or ``None`` if no usable scale
        tag is present.
    """
    for reader in (
        _from_pixel_spacing,
        _from_ultrasound_regions,
        _from_imager_pixel_spacing,
    ):
        try:
            value = reader(ds)
        except Exception as exc:  # never let a malformed tag crash calibration
            logger.debug("DICOM scale reader %s failed: %s", reader.__name__, exc)
            value = None
        if value is not None and value > 0:
            logger.debug("DICOM mm/px from %s: %.5f", reader.__name__, value)
            return value
    return None


def _mean_positive(a: Optional[float], b: Optional[float]) -> Optional[float]:
    vals = [v for v in (a, b) if v is not None and v > 0]
    return sum(vals) / len(vals) if vals else None


def _from_pixel_spacing(ds) -> Optional[float]:
    ps = getattr(ds, "PixelSpacing", None)
    if ps is None:
        return None
    # [row_spacing, col_spacing] in millimetres.
    return _mean_positive(float(ps[0]), float(ps[1]))


def _from_imager_pixel_spacing(ds) -> Optional[float]:
    ps = getattr(ds, "ImagerPixelSpacing", None)
    if ps is None:
        return None
    return _mean_positive(float(ps[0]), float(ps[1]))


def _from_ultrasound_regions(ds) -> Optional[float]:
    regions = getattr(ds, "SequenceOfUltrasoundRegions", None)
    if not regions:
        return None
    for region in regions:
        dx = getattr(region, "PhysicalDeltaX", None)
        dy = getattr(region, "PhysicalDeltaY", None)
        unit_x = getattr(region, "PhysicalUnitsXDirection", None)
        # PhysicalDelta is usually in cm/pixel (unit code 3) -> x10 for mm.
        factor = 10.0 if unit_x == _UNIT_CM else 1.0
        value = _mean_positive(
            abs(float(dx)) * factor if dx is not None else None,
            abs(float(dy)) * factor if dy is not None else None,
        )
        if value is not None:
            return value
    return None
