"""Ноги не видны: восстановление положения человека по макушке.

Когда ноги закрыты (теннисный стол, стойка, обрезаны краем кадра), низ маски — не пол,
и проекция уносит человека от камеры на метр-два. Поза камеры известна, поэтому:
  луч из камеры через верх головы ∩ плоскость на высоте роста (1.70 м) = точка над ногами.
Ошибка роста ±0.1 м даёт ошибку по горизонтали ≈ 0.1 · (расстояние / высота камеры над головой) —
она идёт в σ.

Признак «ноги закрыты»:
  - человек обрезан нижним краем кадра;
  - низ маски попал в проекцию предмета-препятствия (стол) на кадре;
  - «видимый рост» (высота макушки над точкой ног) меньше min_height_m — ноги срезаны чем-то.
Сидящие (видимый рост 1.0–1.45 м, ноги не закрыты) — точка ног верная, только uncertain.
"""
from __future__ import annotations

import numpy as np

from . import polygons as P


def _pose(cam):
    p = cam.pose()
    if p is None:
        raise ValueError(f"{cam.camera_id}: нет позы — восстановление по макушке невозможно")
    return p


def head_to_floor(cam, uv_top, person_h: float = 1.70):
    """Пиксель макушки -> точка пола под ней, при росте person_h. -> (xy (N,2), ok (N,), dist (N,))."""
    R, C = _pose(cam)
    b = cam.distortion.pixels_to_bearings(np.asarray(uv_top, float).reshape(-1, 2))
    d = b @ R
    Z = -person_h
    dz = np.where(np.abs(d[:, 2]) < 1e-9, 1e-9, d[:, 2])
    s = (Z - C[2]) / dz
    ok = (s > 0) & (C[2] < Z)                    # камера выше макушки
    X = C + s[:, None] * d
    out = np.where(ok[:, None], X[:, :2], np.nan)
    return out, ok, np.linalg.norm(out - C[:2], axis=1)


def visible_height(cam, uv_foot, uv_top):
    """Высота макушки над точкой ног, м (по позе камеры). NaN — не посчитать."""
    R, C = _pose(cam)
    xy, ok = cam.project(np.asarray(uv_foot, float).reshape(-1, 2))
    b = cam.distortion.pixels_to_bearings(np.asarray(uv_top, float).reshape(-1, 2))
    d = b @ R
    # Точка на луче макушки, ближайшая по горизонтали к точке ног.
    dh = d[:, :2]
    t = np.einsum("ij,ij->i", xy - C[:2], dh) / np.maximum(np.einsum("ij,ij->i", dh, dh), 1e-12)
    Zt = C[2] + t * d[:, 2]
    h = -Zt
    return np.where(ok & (t > 0), h, np.nan)


def object_image_polygon(cam, obj: dict) -> np.ndarray:
    """Предмет на плане (прямоугольник center_m, size_m, angle_deg, height_m) -> выпуклый контур на кадре."""
    import cv2
    R, C = _pose(cam)
    L, W = obj["size_m"]
    a = np.deg2rad(obj.get("angle_deg", 0.0))
    loc = np.array([[-L / 2, -W / 2], [L / 2, -W / 2], [L / 2, W / 2], [-L / 2, W / 2]])
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    xy = loc @ rot.T + np.asarray(obj["center_m"], float)
    pts = np.vstack([np.column_stack([xy, np.zeros(4)]), np.column_stack([xy, np.full(4, -obj["height_m"])])])
    c = (pts - C) @ R.T
    if (c[:, 2] <= 0).any():
        return np.zeros((0, 2))
    uv = cam.distortion.bearings_to_pixels(c / np.linalg.norm(c, axis=1, keepdims=True))
    return cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2).astype(float)


def recover(cam, uv_foot, uv_top, cut_bottom=None, objects=(), person_h: float = 1.70,
            min_height_m: float = 1.45, seated_min_m: float = 0.95, h_sigma_m: float = 0.10):
    """-> dict: xy (N,2), src ('foot'|'head'|'seated'), extra_sigma (N,), vis_h (N,).

    'head'   — ноги закрыты, положение по макушке;
    'seated' — видимый рост как у сидящего, ноги не закрыты: точка ног, но с запасом σ;
    'foot'   — обычный случай.
    """
    uv_foot = np.asarray(uv_foot, float).reshape(-1, 2)
    uv_top = np.asarray(uv_top, float).reshape(-1, 2)
    n = len(uv_foot)
    cut = np.zeros(n, bool) if cut_bottom is None else np.asarray(cut_bottom, bool)
    xy, _ = cam.project(uv_foot)
    in_obj = np.zeros(n, bool)
    for o in objects:
        poly = object_image_polygon(cam, o)
        if len(poly) >= 3:
            in_obj |= P.contains_tol(poly, uv_foot, 6.0)
    vh = visible_height(cam, uv_foot, uv_top)
    short = np.isfinite(vh) & (vh < seated_min_m)
    occluded = cut | in_obj | short
    hxy, hok, dist = head_to_floor(cam, uv_top, person_h)
    _, C = _pose(cam)
    use_head = occluded & hok
    src = np.full(n, "foot", dtype=object)
    src[use_head] = "head"
    seated = ~occluded & np.isfinite(vh) & (vh < min_height_m)
    src[seated] = "seated"
    extra = np.zeros(n)
    above = max(-C[2] - person_h, 0.3)
    extra[use_head] = h_sigma_m * dist[use_head] / above
    extra[seated] = 0.15
    out = xy.copy()
    out[use_head] = hxy[use_head]
    return {"xy": out, "src": src, "extra_sigma": extra, "vis_h": vh, "occluded": occluded}
