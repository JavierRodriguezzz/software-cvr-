"""Tests for DICOM physical-pixel-size extraction (Phase 2 calibration)."""

import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from core.calibration.dicom_scale import pixel_spacing_mm_from_dataset


def test_pixel_spacing_tag():
    ds = Dataset()
    ds.PixelSpacing = [0.087975, 0.087975]
    assert pixel_spacing_mm_from_dataset(ds) == pytest.approx(0.087975)


def test_ultrasound_regions_cm_converted_to_mm():
    region = Dataset()
    region.PhysicalDeltaX = 0.0088  # cm/pixel
    region.PhysicalDeltaY = 0.0088
    region.PhysicalUnitsXDirection = 3  # code 3 == cm
    ds = Dataset()
    ds.SequenceOfUltrasoundRegions = Sequence([region])
    assert pixel_spacing_mm_from_dataset(ds) == pytest.approx(0.088)


def test_imager_pixel_spacing_fallback():
    ds = Dataset()
    ds.ImagerPixelSpacing = [0.2, 0.2]
    assert pixel_spacing_mm_from_dataset(ds) == pytest.approx(0.2)


def test_pixel_spacing_preferred_over_regions():
    region = Dataset()
    region.PhysicalDeltaX = 0.5
    region.PhysicalUnitsXDirection = 3
    ds = Dataset()
    ds.PixelSpacing = [0.09, 0.09]
    ds.SequenceOfUltrasoundRegions = Sequence([region])
    assert pixel_spacing_mm_from_dataset(ds) == pytest.approx(0.09)


def test_no_scale_returns_none():
    assert pixel_spacing_mm_from_dataset(Dataset()) is None


def test_zero_spacing_ignored():
    ds = Dataset()
    ds.PixelSpacing = [0.0, 0.0]
    assert pixel_spacing_mm_from_dataset(ds) is None
