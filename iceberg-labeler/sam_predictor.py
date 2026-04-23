"""
SAM (Segment Anything Model) service for interactive segmentation.

Provides click-to-segment functionality: a labeler clicks a point on the
chip image, and SAM returns a polygon outlining the object at that location.

Uses MobileSAM by default (9.66M params, ~40MB checkpoint) for fast CPU
inference on 256x256 chips. Can be swapped for larger SAM variants on GPU.

Embedding cache: The image encoder is the expensive step. Once an image is
embedded, subsequent point prompts reuse the cached embedding (~50ms each).
"""

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports — these are heavy and optional
_torch = None
_cv2 = None
_sam_model_registry = None
_SamPredictor = None


def _lazy_imports():
    """Import torch/cv2/SAM on first use so the app starts fast without them."""
    global _torch, _cv2, _sam_model_registry, _SamPredictor

    if _torch is not None:
        return True

    try:
        import torch
        import cv2
        _torch = torch
        _cv2 = cv2
    except ImportError as e:
        logger.warning("SAM dependencies not installed: %s", e)
        return False

    # Try MobileSAM first, fall back to regular SAM
    try:
        from mobile_sam import sam_model_registry, SamPredictor
        _sam_model_registry = sam_model_registry
        _SamPredictor = SamPredictor
        logger.info("Using MobileSAM")
    except ImportError:
        try:
            from segment_anything import sam_model_registry, SamPredictor
            _sam_model_registry = sam_model_registry
            _SamPredictor = SamPredictor
            logger.info("Using segment-anything")
        except ImportError as e:
            logger.warning("No SAM library found: %s", e)
            return False

    return True


class SAMService:
    """
    Singleton service that loads a SAM model and provides predictions.

    Usage:
        sam = SAMService("models/mobile_sam.pt", model_type="vit_t")
        polygon = sam.predict(chip_id=1, image_path="path.png",
                              point_coords=[[128, 100]], point_labels=[1])
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "vit_t",
        device: str = "cpu",
        cache_size: int = 20,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.model_type = model_type
        self.device = device
        self.cache_size = cache_size

        self._predictor: Optional[object] = None
        self._loaded = False
        self._load_error: Optional[str] = None

        # LRU cache: chip_id → (image_path, was_set)
        self._embedding_cache: OrderedDict[int, str] = OrderedDict()
        self._current_chip_id: Optional[int] = None

    @property
    def is_available(self) -> bool:
        """True if SAM model is loaded and ready for predictions."""
        return self._loaded

    @property
    def model_info(self) -> dict:
        """Return model metadata for the /status endpoint."""
        return {
            "available": self._loaded,
            "model_type": self.model_type,
            "device": self.device,
            "checkpoint": str(self.checkpoint_path),
            "cache_size": self.cache_size,
            "cached_chips": list(self._embedding_cache.keys()),
            "error": self._load_error,
        }

    def load(self) -> bool:
        """Load the SAM model. Returns True on success."""
        if not _lazy_imports():
            self._load_error = "SAM dependencies not installed (torch, cv2, mobile_sam)"
            logger.error(self._load_error)
            return False

        if not self.checkpoint_path.exists():
            self._load_error = f"Checkpoint not found: {self.checkpoint_path}"
            logger.error(self._load_error)
            return False

        try:
            sam = _sam_model_registry[self.model_type](
                checkpoint=str(self.checkpoint_path)
            )
            sam.to(device=self.device)
            self._predictor = _SamPredictor(sam)
            self._loaded = True
            self._load_error = None
            logger.info(
                "SAM loaded: type=%s device=%s checkpoint=%s",
                self.model_type, self.device, self.checkpoint_path,
            )
            return True
        except Exception as e:
            self._load_error = str(e)
            logger.error("Failed to load SAM: %s", e)
            return False

    def _set_image(self, chip_id: int, image_path: str) -> None:
        """
        Set the image for SAM prediction, using cache when possible.
        The image encoder runs once per chip; subsequent clicks are cheap.
        """
        # Already embedded?
        if self._current_chip_id == chip_id and chip_id in self._embedding_cache:
            return

        # Load image as RGB numpy array
        image = _cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Convert BGR → RGB (OpenCV loads as BGR)
        image_rgb = _cv2.cvtColor(image, _cv2.COLOR_BGR2RGB)

        # Run image encoder (expensive — ~100-200ms on CPU for 256x256)
        self._predictor.set_image(image_rgb)
        self._current_chip_id = chip_id

        # Update LRU cache
        if chip_id in self._embedding_cache:
            self._embedding_cache.move_to_end(chip_id)
        else:
            self._embedding_cache[chip_id] = image_path
            # Evict oldest if over limit
            while len(self._embedding_cache) > self.cache_size:
                self._embedding_cache.popitem(last=False)

    def predict(
        self,
        chip_id: int,
        image_path: str,
        point_coords: list[list[float]],
        point_labels: list[int],
        multimask: bool = False,
    ) -> dict:
        """
        Run SAM prediction for given click points.

        Parameters
        ----------
        chip_id      : ID of the chip (for embedding cache)
        image_path   : Path to the PNG image file
        point_coords : List of [col, row] pixel coordinates
        point_labels : List of labels (1=foreground, 0=background)
        multimask    : If True, return multiple mask options

        Returns
        -------
        dict with keys:
            polygon    : [[col, row], ...] or [] if no valid mask
            confidence : float (0-1)
            area_px    : float (polygon area in pixels)
        """
        if not self._loaded:
            raise RuntimeError("SAM model not loaded")

        # Embed the image (cached if same chip_id)
        self._set_image(chip_id, image_path)

        # Convert to numpy arrays
        coords_np = np.array(point_coords, dtype=np.float32)  # shape (N, 2): [col, row]
        labels_np = np.array(point_labels, dtype=np.int32)     # shape (N,)

        # SAM expects (x, y) = (col, row) — our convention matches
        masks, scores, logits = self._predictor.predict(
            point_coords=coords_np,
            point_labels=labels_np,
            multimask_output=multimask,
        )

        # Pick the best mask (highest score)
        if len(masks) == 0:
            return {"polygon": [], "confidence": 0.0, "area_px": 0.0}

        best_idx = int(np.argmax(scores))
        mask = masks[best_idx]
        confidence = float(scores[best_idx])

        # Convert binary mask → polygon
        polygon = self._mask_to_polygon(mask)
        area_px = self._polygon_area(polygon) if polygon else 0.0

        return {
            "polygon": polygon,
            "confidence": round(confidence, 4),
            "area_px": round(area_px, 2),
        }

    def _mask_to_polygon(
        self, mask: np.ndarray, epsilon_pct: float = 1.0
    ) -> list[list[float]]:
        """
        Convert a binary mask to a simplified polygon.

        Parameters
        ----------
        mask        : 2D binary array (H, W) of 0s and 1s
        epsilon_pct : Polygon simplification factor (% of perimeter)

        Returns
        -------
        List of [col, row] coordinates, or [] if no valid contour.
        """
        mask_uint8 = (mask * 255).astype(np.uint8)

        contours, _ = _cv2.findContours(
            mask_uint8, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return []

        # Use the largest contour
        contour = max(contours, key=_cv2.contourArea)

        # Simplify polygon
        perimeter = _cv2.arcLength(contour, True)
        epsilon = epsilon_pct * perimeter / 100.0
        approx = _cv2.approxPolyDP(contour, epsilon, True)

        # Convert to [col, row] format
        polygon = approx.reshape(-1, 2).tolist()

        # Need at least 3 points
        if len(polygon) < 3:
            return []

        # Round to 2 decimal places
        polygon = [[round(col, 2), round(row, 2)] for col, row in polygon]

        return polygon

    @staticmethod
    def _polygon_area(coords: list[list[float]]) -> float:
        """Shoelace formula for polygon area."""
        n = len(coords)
        if n < 3:
            return 0.0
        area = 0.0
        for i in range(n):
            x1, y1 = coords[i]
            x2, y2 = coords[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0

    def clear_cache(self, chip_id: Optional[int] = None) -> None:
        """Clear embedding cache for a specific chip, or all chips."""
        if chip_id is not None:
            self._embedding_cache.pop(chip_id, None)
            if self._current_chip_id == chip_id:
                self._current_chip_id = None
        else:
            self._embedding_cache.clear()
            self._current_chip_id = None


# ── Module-level singleton ─────────────────────────────────────────────────────

_sam_service: Optional[SAMService] = None


def get_sam_service() -> Optional[SAMService]:
    """Return the global SAM service instance, or None if not initialized."""
    return _sam_service


def init_sam_service(
    checkpoint_path: str,
    model_type: str = "vit_t",
    device: str = "cpu",
    cache_size: int = 20,
) -> Optional[SAMService]:
    """
    Initialize the global SAM service. Returns None if loading fails.
    Safe to call if dependencies are missing — will log a warning and return None.
    """
    global _sam_service

    service = SAMService(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        device=device,
        cache_size=cache_size,
    )

    if service.load():
        _sam_service = service
        return service
    else:
        logger.warning("SAM service not available: %s", service._load_error)
        _sam_service = service  # Keep it so /status can report the error
        return None
