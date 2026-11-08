"""Растровая карта зон плана: какие камеры видят точку, в какой она зоне.

Растр с шагом 0.1 м: запрос по сотням тысяч точек — одна индексация массива.
Главная зона по приоритету: exit > portal > occluder > overlap > open.
"""
from __future__ import annotations

import cv2
import numpy as np

ZONE_PRIORITY = ("exit", "portal", "occluder", "overlap", "open")


def rasterize(polys, shape, res, origin=(0.0, 0.0)) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for p in polys:
        q = (np.asarray(p, float) - origin) / res
        cv2.fillPoly(m, [np.round(q * 8).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=3)
    return m.astype(bool)


class ZoneMap:
    def __init__(self, plan, fovs: dict, res: float = 0.1, portals=None):
        """plan — Plan; fovs — {camera_id: полигон зоны обзора на плане, метры};
        portals — {gap_id: полигон портала} (из graph.json)."""
        self.res = res
        W, H = plan.size_m
        self.shape = (int(np.ceil(H / res)) + 1, int(np.ceil(W / res)) + 1)
        self.cam_ids = sorted(fovs)
        self.floor = rasterize(plan.floor, self.shape, res)
        self.cover = np.stack([rasterize([fovs[c]], self.shape, res) & self.floor for c in self.cam_ids]) \
            if self.cam_ids else np.zeros((0, *self.shape), bool)
        self.exit_ids = [e["id"] for e in plan.exits]
        self.exit_idx = np.full(self.shape, -1, np.int16)
        for i, e in enumerate(plan.exits):
            self.exit_idx[rasterize([e["polygon"]], self.shape, res)] = i
        self.occ = rasterize([o["polygon"] for o in plan.occluders], self.shape, res) \
            if plan.occluders else np.zeros(self.shape, bool)
        self.portal = np.zeros(self.shape, bool)
        for p in (portals or {}).values():
            self.portal |= rasterize([p], self.shape, res)

    def _ij(self, pts):
        pts = np.asarray(pts, float).reshape(-1, 2)
        fin = np.isfinite(pts).all(1)
        ij = np.zeros((len(pts), 2), int)
        ij[fin] = np.round(pts[fin, ::-1] / self.res).astype(int)
        inside = fin & (ij[:, 0] >= 0) & (ij[:, 0] < self.shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < self.shape[1])
        ij[~inside] = 0
        return ij, inside

    def cams_cover(self, pts) -> list:
        ij, ok = self._ij(pts)
        if not self.cam_ids:
            return [[] for _ in range(len(ij))]
        c = self.cover[:, ij[:, 0], ij[:, 1]] & ok[None]
        return [[self.cam_ids[k] for k in np.flatnonzero(c[:, i])] for i in range(len(ij))]

    def n_cover(self, pts) -> np.ndarray:
        ij, ok = self._ij(pts)
        if not self.cam_ids:
            return np.zeros(len(ij), int)
        return (self.cover[:, ij[:, 0], ij[:, 1]] & ok[None]).sum(0)

    def zone(self, pts) -> np.ndarray:
        """Главная зона: 'exit:<id>' | 'portal' | 'occluder' | 'overlap' | 'open' | 'outside'."""
        ij, ok = self._ij(pts)
        i, j = ij[:, 0], ij[:, 1]
        out = np.full(len(ij), "open", dtype=object)
        ncov = self.n_cover(pts)
        out[ncov >= 2] = "overlap"
        out[self.occ[i, j]] = "occluder"
        out[self.portal[i, j]] = "portal"
        ex = self.exit_idx[i, j]
        for k in np.flatnonzero(ex >= 0):
            out[k] = "exit:" + self.exit_ids[ex[k]]
        out[~ok | ~self.floor[i, j] & (ex < 0)] = "outside"
        return out


def visible_floor(cam, plan, res: float = 0.1, max_dist_m: float = 14.0, max_theta_deg: float = 85.0,
                  los_step_m: float = 0.1) -> np.ndarray:
    """Растр пола, который камера видит: в кадре, не дальше max_dist, прямая видимость по полу плана
    (луч от проекции камеры на пол не выходит за пол — значит, не сквозь стену)."""
    W, H = plan.size_m
    shape = (int(np.ceil(H / res)) + 1, int(np.ceil(W / res)) + 1)
    floor = rasterize(plan.floor, shape, res)
    pose = cam.pose()
    if pose is None:
        raise ValueError(f"{cam.camera_id}: нет позы камеры")
    R, C = pose
    ij = np.argwhere(floor)
    XY = ij[:, ::-1] * res
    d = np.linalg.norm(XY - C[:2], axis=1)
    keep = d < max_dist_m
    ij, XY = ij[keep], XY[keep]
    uv, ok = cam.unproject(XY)
    w, h = cam.image_size
    c = (np.column_stack([XY, np.zeros(len(XY))]) - C) @ R.T
    th = np.degrees(np.arccos(np.clip(c[:, 2] / np.linalg.norm(c, axis=1), -1, 1)))
    ok &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h) & (th < max_theta_deg)
    ij, XY = ij[ok], XY[ok]
    # Прямая видимость по полу. Старт луча — ближайшая к камере точка пола.
    fl = np.argwhere(floor)[:, ::-1] * res
    start = fl[np.argmin(np.linalg.norm(fl - C[:2], axis=1))]
    L = np.linalg.norm(XY - start, axis=1)
    n = int(np.ceil(L.max() / los_step_m)) + 1 if len(L) else 1
    vis = np.ones(len(XY), bool)
    for t in np.linspace(0, 1, n):
        P = start + t * (XY - start)
        q = np.round(P[:, ::-1] / res).astype(int)
        q[:, 0] = q[:, 0].clip(0, shape[0] - 1); q[:, 1] = q[:, 1].clip(0, shape[1] - 1)
        # Допуск: 1 клетка от пола (люди у стены, неточный полигон).
        vis &= floor_grow(floor)[q[:, 0], q[:, 1]]
    out = np.zeros(shape, bool)
    out[ij[vis, 0], ij[vis, 1]] = True
    return out


_GROW_CACHE = {}


def floor_grow(floor):
    key = id(floor)
    if key not in _GROW_CACHE:
        _GROW_CACHE.clear()
        _GROW_CACHE[key] = cv2.dilate(floor.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return _GROW_CACHE[key]


def raster_to_polygon(mask, res: float, simplify_m: float = 0.15):
    """Самый большой контур растра -> полигон в метрах."""
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return []
    c = max(cs, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, simplify_m / res, True).reshape(-1, 2)
    return (c * res).round(3).tolist()
