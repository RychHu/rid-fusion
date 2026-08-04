"""Small dependency-free metrics used by tests and reports."""

from __future__ import annotations

import math
import numpy as np


def horizontal_error_m(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    mean_lat = math.radians((lat_a + lat_b) / 2.0)
    dy = (lat_b - lat_a) * 111_320.0
    dx = (lon_b - lon_a) * 111_320.0 * max(math.cos(mean_lat), 1e-3)
    return math.hypot(dx, dy)


def rmse(values) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.sqrt(np.mean(array * array))) if array.size else 0.0


def percentile(values, q: float) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.percentile(array, q)) if array.size else 0.0
