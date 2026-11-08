"""COCO RLE без pycocotools: кодирование, декодирование и низ маски прямо по RLE.

Формат как у pycocotools: {"size": [H, W], "counts": "<строка>"}, маска развёрнута
по столбцам (Fortran order), counts чередуют 0 и 1, начиная с нулей.
В детекциях маска хранится строкой JSON (см. контракт detections.parquet).
"""
from __future__ import annotations

import json

import numpy as np


# ------------------------------------------------------------ строка counts <-> массив

def _counts_from_string(s: str) -> np.ndarray:
    cnts, p, m = [], 0, 0
    b = s.encode("ascii") if isinstance(s, str) else s
    while p < len(b):
        x, k, more = 0, 0, True
        while more:
            c = b[p] - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if m > 2:
            x += cnts[m - 2]
        cnts.append(x)
        m += 1
    return np.asarray(cnts, dtype=np.int64)


def _counts_to_string(cnts) -> str:
    out = []
    cnts = [int(c) for c in cnts]
    for i, x in enumerate(cnts):
        if i > 2:
            x -= cnts[i - 2]
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)


def parse(rle) -> tuple[int, int, np.ndarray]:
    """str (JSON) | dict -> (H, W, counts)."""
    if isinstance(rle, str):
        rle = json.loads(rle)
    H, W = int(rle["size"][0]), int(rle["size"][1])
    c = rle["counts"]
    counts = np.asarray(c, dtype=np.int64) if isinstance(c, list) else _counts_from_string(c)
    return H, W, counts


def encode(mask: np.ndarray) -> dict:
    m = np.asarray(mask, dtype=bool)
    H, W = m.shape
    flat = m.ravel(order="F").astype(np.int8)
    change = np.flatnonzero(np.diff(flat)) + 1
    edges = np.r_[0, change, flat.size]
    runs = np.diff(edges)
    if flat.size and flat[0] == 1:
        runs = np.r_[0, runs]
    return {"size": [H, W], "counts": _counts_to_string(runs)}


def decode(rle) -> np.ndarray:
    H, W, counts = parse(rle)
    vals = np.zeros(len(counts), dtype=bool)
    vals[1::2] = True
    flat = np.repeat(vals, counts)
    if flat.size < H * W:
        flat = np.r_[flat, np.zeros(H * W - flat.size, bool)]
    return flat[:H * W].reshape((H, W), order="F")


# ------------------------------------------------------------ низ маски без полного декода

def column_segments(rle):
    """Отрезки единиц по столбцам: (col, row_start, row_end) включительно. Без декода кадра 5 Мпикс."""
    H, W, counts = parse(rle)
    starts = np.cumsum(np.r_[0, counts[:-1]])
    ones = np.arange(len(counts)) % 2 == 1
    s, L = starts[ones], counts[ones]
    keep = L > 0
    s, e = s[keep], s[keep] + L[keep] - 1
    c0, c1 = s // H, e // H
    n = (c1 - c0 + 1)
    col = np.repeat(c0, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    first = np.r_[True, np.diff(np.repeat(np.arange(len(s)), n)) != 0]
    last = np.r_[first[1:], True]
    rs = np.where(first, np.repeat(s % H, n), 0)
    re = np.where(last, np.repeat(e % H, n), H - 1)
    return H, W, col, rs, re


def bottom_info(rle, rows: int = 4):
    """Низ маски: (y_bottom, x_median по нижним `rows` строкам, y_top, area). None — пустая маска."""
    H, W, col, rs, re = column_segments(rle)
    if len(col) == 0:
        return None
    yb = int(re.max())
    lo = yb - rows + 1
    ov = np.minimum(re, yb) - np.maximum(rs, lo) + 1          # сколько пикселей отрезка в нижних строках
    sel = ov > 0
    xs, w = col[sel].astype(float), ov[sel].astype(float)
    order = np.argsort(xs)
    xs, w = xs[order], w[order]
    cw = np.cumsum(w)
    xm = float(xs[np.searchsorted(cw, cw[-1] / 2)])
    return yb, xm, int(rs.min()), int((re - rs + 1).sum())


def top_info(rle, rows: int = 4):
    """Верх маски: (y_top, x_median по верхним `rows` строкам). None — пустая маска."""
    H, W, col, rs, re = column_segments(rle)
    if len(col) == 0:
        return None
    yt = int(rs.min())
    hi = yt + rows - 1
    ov = np.minimum(re, hi) - np.maximum(rs, yt) + 1
    sel = ov > 0
    xs, w = col[sel].astype(float), ov[sel].astype(float)
    order = np.argsort(xs)
    xs, w = xs[order], w[order]
    cw = np.cumsum(w)
    return yt, float(xs[np.searchsorted(cw, cw[-1] / 2)])
