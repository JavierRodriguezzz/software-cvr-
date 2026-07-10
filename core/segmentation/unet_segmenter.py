"""CervixAI V2 — U-Net segmentation backend.

Wraps the existing UNetSegmenter from models/unet_model.py without
modifying it.  Implements ISegmenter so the pipeline depends only on
the interface, not on the concrete U-Net class.

The model is injected at construction time (Dependency Injection).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from core.domain import SegmentationResult
from core.interfaces import ISegmenter

logger = logging.getLogger(__name__)


class UNetSegmenter(ISegmenter):
    """ISegmenter implementation wrapping the existing U-Net model.

    The underlying ``UNet`` architecture and weights are loaded from
    ``models/unet_model.py`` without any modification, preserving the
    exact inference behaviour.

    Args:
        model_path: Path to the ``.pth`` weights file.
        device: Torch device.  Auto-detected (CUDA > CPU) if ``None``.
    """

    def __init__(self, model_path: Path, device: Optional[torch.device] = None) -> None:
        """Load the U-Net weights and prepare for inference."""
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model_path = Path(model_path)
        self._segmenter = self._load_model()

    def _load_model(self):
        """Dynamically import and instantiate the existing UNetSegmenter.

        Returns:
            An instance of the legacy ``UNetSegmenter`` class.

        Raises:
            RuntimeError: If the weights file or module cannot be loaded.
        """
        import importlib.util

        module_path = self._model_path.parent / "unet_model.py"
        if not module_path.exists():
            raise RuntimeError(f"unet_model.py not found at {module_path}")
        if not self._model_path.exists():
            raise RuntimeError(f"U-Net weights not found at {self._model_path}")

        spec = importlib.util.spec_from_file_location("unet_model", str(module_path))
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to create module spec for unet_model.py")

        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        segmenter = mod.UNetSegmenter(self._model_path, self._device)
        logger.info(
            "U-Net loaded from %s on %s", self._model_path, self._device
        )
        return segmenter

    @property
    def backend_name(self) -> str:
        """Return the backend identifier."""
        return "unet"

    def predict(self, image_bgr: np.ndarray) -> SegmentationResult:
        """Run U-Net inference on a BGR image.

        The raw ``UNetSegmenter.predict`` is called unchanged.
        Probability maps are not exposed by the legacy model, so
        ``probability_map`` is set to ``None``.

        Args:
            image_bgr: Input image in BGR colour order, shape (H, W, 3).

        Returns:
            SegmentationResult containing the integer prediction mask.
        """
        t0 = time.perf_counter()
        prediction = self._segmenter.predict(image_bgr)
        elapsed = time.perf_counter() - t0

        logger.debug(
            "U-Net inference completed in %.3fs — unique classes: %s",
            elapsed,
            np.unique(prediction).tolist(),
        )
        return SegmentationResult(
            prediction=prediction,
            probability_map=None,
            model_backend=self.backend_name,
            execution_time=elapsed,
        )
