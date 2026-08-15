"""Unicode-safe OpenCV image loading for Windows and POSIX paths."""

from pathlib import Path

import cv2
import numpy as np


def read_color_image(path: Path) -> np.ndarray | None:
    """Read a color image without relying on OpenCV's narrow Windows path API."""
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
