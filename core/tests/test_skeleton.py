"""Tests for SkeletonExtractor and SkeletonGraphBuilder."""

import numpy as np
import pytest
from core.measurement.skeleton_extractor import SkeletonExtractor
from core.measurement.graph_builder import SkeletonGraphBuilder


def _horizontal_bar_mask(h=80, w=200, bar_y=40, bar_h=10):
    """Create a horizontal bar binary mask for testing."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[bar_y:bar_y + bar_h, 10:w - 10] = 255
    return mask


def test_skeleton_returns_thin_structure():
    mask = _horizontal_bar_mask()
    extractor = SkeletonExtractor(min_branch_length=5)
    skel = extractor.extract(mask)
    assert skel is not None
    # Skeleton should be thinner than the original
    assert skel.sum() < mask.sum()


def test_skeleton_not_none_for_valid_mask():
    mask = _horizontal_bar_mask()
    extractor = SkeletonExtractor()
    skel = extractor.extract(mask)
    assert skel is not None


def test_skeleton_returns_none_for_empty_mask():
    mask = np.zeros((100, 100), dtype=np.uint8)
    extractor = SkeletonExtractor()
    skel = extractor.extract(mask)
    assert skel is None


def test_graph_builder_returns_graph():
    mask = _horizontal_bar_mask()
    extractor = SkeletonExtractor(min_branch_length=5)
    skel = extractor.extract(mask)
    assert skel is not None
    builder = SkeletonGraphBuilder()
    graph = builder.build(skel)
    assert graph.number_of_nodes() >= 2


def test_graph_has_weighted_edges():
    mask = _horizontal_bar_mask()
    extractor = SkeletonExtractor(min_branch_length=5)
    skel = extractor.extract(mask)
    builder = SkeletonGraphBuilder()
    graph = builder.build(skel)
    for u, v, data in graph.edges(data=True):
        assert "weight" in data
        assert data["weight"] > 0
