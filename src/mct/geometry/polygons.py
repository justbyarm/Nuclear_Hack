"""Векторизованные операции с полигонами на numpy (без shapely).

Полигон — массив (K, 2) вершин, замыкать не нужно.
"""
from __future__ import annotations

import numpy as np


def _as_poly(poly) -> np.ndarray:
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if len(p) < 3:
        raise ValueError("полигону нужно хотя бы 3 вершины")
    return p


def contains(poly, pts) -> np.ndarray:
    """Точки внутри полигона (ray casting). pts: (N, 2) -> bool (N,)."""
    p = _as_poly(poly)
    q = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    x, y = q[:, 0:1], q[:, 1:2]                      # (N, 1)
    x1, y1 = p[:, 0], p[:, 1]                        # (K,)
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    crosses = (y1 > y) != (y2 > y)                   # (N, K)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    hits = crosses & (x < xint)
    return (hits.sum(axis=1) % 2) == 1


def distance_to_boundary(poly, pts) -> np.ndarray:
    """Расстояние от точек до границы полигона (без знака)."""
    p = _as_poly(poly)
    q = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    a = p[None, :, :]                                # (1, K, 2)
    b = np.roll(p, -1, axis=0)[None, :, :]
    ab = b - a
    ap = q[:, None, :] - a                           # (N, K, 2)
    denom = np.maximum((ab ** 2).sum(-1), 1e-12)
    t = np.clip((ap * ab).sum(-1) / denom, 0.0, 1.0)
    closest = a + t[..., None] * ab
    d = np.linalg.norm(q[:, None, :] - closest, axis=-1)
    return d.min(axis=1)


def signed_distance(poly, pts) -> np.ndarray:
    """> 0 внутри, < 0 снаружи (как cv2.pointPolygonTest)."""
    d = distance_to_boundary(poly, pts)
    return np.where(contains(poly, pts), d, -d)


def contains_tol(poly, pts, tol: float = 0.0) -> np.ndarray:
    """Внутри полигона или снаружи не дальше tol."""
    return signed_distance(poly, pts) >= -tol


def area(poly) -> float:
    p = _as_poly(poly)
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def contains_chunked(poly, pts, tol: float = 0.0, chunk: int = 200_000) -> np.ndarray:
    """То же, что contains_tol, но кусками — для сотен тысяч точек."""
    q = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    out = np.empty(len(q), dtype=bool)
    for i in range(0, len(q), chunk):
        out[i:i + chunk] = contains_tol(poly, q[i:i + chunk], tol)
    return out


def densify(poly, step: float, closed: bool = True) -> np.ndarray:
    """Добавить точки на стороны полигона с шагом step (в единицах вершин)."""
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    nxt = np.roll(p, -1, axis=0) if closed else p[1:]
    cur = p if closed else p[:-1]
    out = []
    for a, b in zip(cur, nxt):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
    if not closed:
        out.append(p[-1:])
    return np.concatenate(out)


def nearest_on_polys(polys, pts) -> np.ndarray:
    """Ближайшая точка внутри/на границе объединения полигонов (точки внутри не двигаются)."""
    q = np.asarray(pts, dtype=np.float64).reshape(-1, 2).copy()
    inside = np.zeros(len(q), bool)
    for p in polys:
        inside |= contains(p, q)
    out = q.copy()
    idx = np.flatnonzero(~inside & np.isfinite(q).all(1))
    if len(idx) == 0:
        return out
    best_d = np.full(len(idx), np.inf)
    best = q[idx].copy()
    for poly in polys:
        p = _as_poly(poly)
        a, b = p, np.roll(p, -1, axis=0)
        ab = b - a
        ap = q[idx][:, None, :] - a[None]
        t = np.clip((ap * ab).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-12), 0, 1)
        c = a[None] + t[..., None] * ab[None]
        d = np.linalg.norm(q[idx][:, None, :] - c, axis=-1)
        k = d.argmin(1)
        dk = d[np.arange(len(idx)), k]
        upd = dk < best_d
        best_d[upd] = dk[upd]
        best[upd] = c[np.arange(len(idx)), k][upd]
    out[idx] = best
    return out
