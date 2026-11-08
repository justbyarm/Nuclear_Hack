"""Точка ног человека на кадре."""
from __future__ import annotations

import numpy as np

from . import rle as R

FOOT_SRC = ("mask", "bbox")


def foot_point(mask_rle, bbox, image_size, bottom_rows: int = 4, cut_margin_px: int = 3):
    """-> (u, v, src, cut_bottom).

    u — медиана x по нижним строкам маски (устойчивее середины рамки к руке и сумке),
    v — нижняя строка маски. Пустая или битая маска — низ-центр рамки.
    cut_bottom — человек обрезан нижним краем кадра: низ маски — не ноги.
    """
    W, H = image_size
    x1, y1, x2, y2 = bbox
    info = None
    if mask_rle is not None and isinstance(mask_rle, (str, dict)) and mask_rle != "":
        try:
            info = R.bottom_info(mask_rle, bottom_rows)
        except (ValueError, KeyError, TypeError, IndexError):
            info = None
    if info is not None:
        v, u, _, _ = info
        src = "mask"
    else:
        u, v, src = (x1 + x2) / 2.0, float(y2), "bbox"
    cut = v >= H - 1 - cut_margin_px
    return float(u), float(v), src, bool(cut)
