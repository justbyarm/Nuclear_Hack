"""Отладочные картинки: сетка 1×1 м на кадре, полигоны, точки."""
from __future__ import annotations

import cv2
import numpy as np

from .camera import Camera


def _polyline_segments(uv, ok, W, H):
    """Режем ломаную на куски там, где точки невалидны или за кадром."""
    inside = ok & np.isfinite(uv).all(1) & (uv[:, 0] >= -W) & (uv[:, 0] <= 2 * W) \
        & (uv[:, 1] >= -H) & (uv[:, 1] <= 2 * H)
    seg, out = [], []
    for p, good in zip(uv, inside):
        if good:
            seg.append(p)
        elif len(seg) > 1:
            out.append(np.array(seg)); seg = []
        else:
            seg = []
    if len(seg) > 1:
        out.append(np.array(seg))
    return out


def draw_grid(img, cam: Camera, bounds_m, step_m: float = 1.0, color=(0, 255, 255),
              thickness: int = 2, floor_polys=None, samples_per_m: int = 20):
    """Сетка плана через обратную проекцию. Линии плотно сэмплируются — под fisheye они кривые.

    bounds_m: (x0, y0, x1, y1) прямоугольник плана в метрах.
    floor_polys: если дан — сетку рисуем только по полу (точки вне пола отбрасываем).
    """
    from . import polygons as P
    out = img.copy()
    H, W = out.shape[:2]
    x0, y0, x1, y1 = bounds_m
    lines = []
    for x in np.arange(np.floor(x0), x1 + 1e-9, step_m):
        ys = np.linspace(y0, y1, max(2, int((y1 - y0) * samples_per_m)))
        lines.append(np.column_stack([np.full_like(ys, x), ys]))
    for y in np.arange(np.floor(y0), y1 + 1e-9, step_m):
        xs = np.linspace(x0, x1, max(2, int((x1 - x0) * samples_per_m)))
        lines.append(np.column_stack([xs, np.full_like(xs, y)]))
    for XY in lines:
        uv, ok = cam.unproject(XY)
        if floor_polys:
            on = np.zeros(len(XY), bool)
            for poly in floor_polys:
                on |= P.contains_tol(poly, XY, 0.05)
            ok = ok & on
        for seg in _polyline_segments(uv, ok, W, H):
            cv2.polylines(out, [np.round(seg).astype(np.int32)], False, color, thickness, cv2.LINE_AA)
    return out


def draw_points(img, pts, color, radius=6, labels=None, thickness=2):
    for i, (u, v) in enumerate(np.asarray(pts, float).reshape(-1, 2)):
        if not np.isfinite([u, v]).all():
            continue
        c = (int(round(u)), int(round(v)))
        cv2.circle(img, c, radius, color, thickness, cv2.LINE_AA)
        if labels is not None:
            cv2.putText(img, str(labels[i]), (c[0] + radius + 2, c[1] - radius - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img


def draw_poly(img, pts, color, closed=True, thickness=2, fill_alpha=0.0):
    p = np.round(np.asarray(pts, float).reshape(-1, 2)).astype(np.int32)
    if len(p) == 0:
        return img
    if fill_alpha > 0 and closed and len(p) >= 3:
        ov = img.copy()
        cv2.fillPoly(ov, [p], color)
        cv2.addWeighted(ov, fill_alpha, img, 1 - fill_alpha, 0, dst=img)
    if len(p) >= 2:
        cv2.polylines(img, [p], closed, color, thickness, cv2.LINE_AA)
    for q in p:
        cv2.circle(img, tuple(int(v) for v in q), 3, color, -1, cv2.LINE_AA)
    return img


def birdseye(cam: Camera, frame, plan_img, m_per_px: float, alpha: float = 0.6, max_dist_m: float = 12.0,
             mask_floor=None):
    """Кадр, спроецированный на план (вид сверху), поверх картинки плана — главный способ
    глазами проверить калибровку: стены и двери кадра должны лечь на стены плана."""
    H, W = plan_img.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W]
    XY = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64) * m_per_px
    pose = cam.pose()
    ok_d = np.ones(len(XY), bool)
    if pose is not None:
        ok_d = np.linalg.norm(XY - pose[1][:2], axis=1) < max_dist_m
    uv = np.full((len(XY), 2), -1.0)
    sub, ok = cam.unproject(XY[ok_d])
    uvs = np.where(ok[:, None], sub, -1.0)
    uv[ok_d] = uvs
    fh, fw = frame.shape[:2]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < fw - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < fh - 1)
    mx = uv[:, 0].reshape(H, W).astype(np.float32)
    my = uv[:, 1].reshape(H, W).astype(np.float32)
    warped = cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    m = inside.reshape(H, W)
    if mask_floor is not None:
        m &= mask_floor
    out = plan_img.copy()
    out[m] = (alpha * warped[m] + (1 - alpha) * plan_img[m]).astype(np.uint8)
    return out
