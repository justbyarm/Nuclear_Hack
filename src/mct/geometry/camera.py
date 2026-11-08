"""Модель камеры: дисторсия «рыбий глаз» + гомография пол кадра → план.

Цепочка для точки на полу:
    пиксель (u, v) --undistort--> идеальные координаты (x, y)  --H--> план (X, Y), метры

Идеальные координаты — нормализованные координаты пинхол-камеры (x = tanθ·cosφ, ...).
Если дисторсии нет (distortion=None), идеальные координаты = пиксели.

Модель дисторсии — эквидистантный fisheye, как в cv2.fisheye:
    θd = θ · (1 + k1 θ² + k2 θ⁴ + k3 θ⁶ + k4 θ⁸),   r_px = f · θd
Своя реализация на numpy: работает и за 90°, совпадает с cv2.fisheye (есть тест).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EPS = 1e-9


# ---------------------------------------------------------------- дисторсия

@dataclass
class Fisheye:
    f: float
    cx: float
    cy: float
    k: tuple = (0.0, 0.0, 0.0, 0.0)   # k1..k4

    def to_dict(self) -> dict:
        return {"model": "fisheye_equidistant", "f": self.f, "cx": self.cx,
                "cy": self.cy, "k": list(self.k)}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Fisheye | None":
        if not d:
            return None
        k = tuple(d.get("k", (0, 0, 0, 0))) + (0.0,) * 4
        return cls(float(d["f"]), float(d["cx"]), float(d["cy"]), tuple(float(v) for v in k[:4]))

    # θ -> θd
    def _poly(self, th):
        k1, k2, k3, k4 = self.k
        t2 = th * th
        return th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))

    def _dpoly(self, th):
        k1, k2, k3, k4 = self.k
        t2 = th * th
        return 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))

    def theta_from_thetad(self, thd, iters: int = 30):
        th = np.array(thd, dtype=np.float64, copy=True)
        for _ in range(iters):
            g = self._poly(th) - thd
            d = self._dpoly(th)
            d = np.where(np.abs(d) < 1e-6, 1e-6, d)
            step = g / d
            th = th - step
            if np.max(np.abs(step), initial=0) < 1e-12:
                break
        return th

    def pixels_to_bearings(self, uv) -> np.ndarray:
        """Пиксели -> единичные лучи (N, 3), z вперёд. Работает и за 90°."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        dx, dy = uv[:, 0] - self.cx, uv[:, 1] - self.cy
        r = np.hypot(dx, dy)
        th = self.theta_from_thetad(r / self.f)
        s = np.where(r > EPS, np.sin(th) / np.maximum(r, EPS), 1.0 / self.f)
        return np.stack([dx * s, dy * s, np.cos(th)], axis=1)

    def bearings_to_pixels(self, b) -> np.ndarray:
        b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
        rxy = np.hypot(b[:, 0], b[:, 1])
        th = np.arctan2(rxy, b[:, 2])
        rd = self.f * self._poly(th)
        s = np.where(rxy > EPS, rd / np.maximum(rxy, EPS), self.f)
        return np.stack([self.cx + b[:, 0] * s, self.cy + b[:, 1] * s], axis=1)

    def undistort(self, uv):
        """Пиксели -> идеальные (x, y) и маска valid (луч смотрит вперёд, θ < 90°)."""
        b = self.pixels_to_bearings(uv)
        valid = b[:, 2] > 1e-3
        z = np.where(valid, b[:, 2], 1.0)
        return b[:, :2] / z[:, None], valid

    def distort(self, xy) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        return self.bearings_to_pixels(np.column_stack([xy, np.ones(len(xy))]))


# ---------------------------------------------------------------- камера

@dataclass
class Camera:
    camera_id: str
    image_size: tuple                      # (W, H) исходного кадра
    H: np.ndarray | None = None            # идеальные -> план, метры
    distortion: Fisheye | None = None
    raw: dict = field(default_factory=dict)   # весь json калибровки как есть

    # --- пиксели <-> идеальные
    def to_ideal(self, uv):
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        if self.distortion is None:
            return uv.copy(), np.ones(len(uv), dtype=bool)
        return self.distortion.undistort(uv)

    def from_ideal(self, xy):
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        if self.distortion is None:
            return xy.copy()
        return self.distortion.distort(xy)

    # --- поза (если есть): работает и для лучей за 90° от оси объектива,
    # которые гомография через пинхол-координаты представить не может.
    MAX_THETA = np.deg2rad(110)

    def pose(self):
        p = self.raw.get("pose") if self.raw else None
        if not p or self.distortion is None or "rvec" not in p or "center_m" not in p:
            return None
        import cv2
        R = cv2.Rodrigues(np.asarray(p["rvec"], dtype=np.float64))[0]
        return R, np.asarray(p["center_m"], dtype=np.float64)

    # --- кадр -> план
    def project(self, uv):
        """Пиксели исходного кадра -> (XY метры (N, 2), valid (N,)).

        valid=False: луч не попадает на пол перед камерой (выше горизонта)
        или точка слишком далеко от оси объектива. Координаты таких точек — NaN.
        """
        pose = self.pose()
        if pose is not None:
            R, C = pose
            uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
            b = self.distortion.pixels_to_bearings(uv)
            d = b @ R
            dz = np.where(np.abs(d[:, 2]) < 1e-12, 1e-12, d[:, 2])
            s = -C[2] / dz
            valid = (s > 0) & (np.arccos(np.clip(b[:, 2], -1, 1)) < self.MAX_THETA)
            out = np.full((len(uv), 2), np.nan)
            out[valid] = C[:2] + s[valid, None] * d[valid, :2]
            return out, valid
        if self.H is None:
            raise RuntimeError(f"{self.camera_id}: гомография не посчитана")
        xy, ok = self.to_ideal(uv)
        ph = np.column_stack([xy, np.ones(len(xy))]) @ self.H.T
        w = ph[:, 2]
        valid = ok & (w > 1e-9)
        out = np.full((len(xy), 2), np.nan)
        out[valid] = ph[valid, :2] / w[valid, None]
        return out, valid

    def project_one(self, u, v):
        p, ok = self.project([[u, v]])
        return (float(p[0, 0]), float(p[0, 1])) if ok[0] else None

    # --- план -> кадр (для сетки и отрисовки)
    def unproject(self, XY):
        pose = self.pose()
        if pose is not None:
            R, C = pose
            XY = np.asarray(XY, dtype=np.float64).reshape(-1, 2)
            c = (np.column_stack([XY, np.zeros(len(XY))]) - C) @ R.T
            b = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
            valid = np.arccos(np.clip(b[:, 2], -1, 1)) < self.MAX_THETA
            out = np.full((len(XY), 2), np.nan)
            if valid.any():
                out[valid] = self.distortion.bearings_to_pixels(b[valid])
            return out, valid
        if self.H is None:
            raise RuntimeError(f"{self.camera_id}: гомография не посчитана")
        XY = np.asarray(XY, dtype=np.float64).reshape(-1, 2)
        ph = np.column_stack([XY, np.ones(len(XY))]) @ np.linalg.inv(self.H).T
        c = ph[:, 2]
        valid = c > 1e-9
        out = np.full((len(XY), 2), np.nan)
        if valid.any():
            out[valid] = self.from_ideal(ph[valid, :2] / c[valid, None])
        return out, valid

    def meters_per_pixel(self, uv) -> np.ndarray:
        """Сколько метров плана приходится на 1 пиксель кадра в точке (макс. по осям)."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        p0, v0 = self.project(uv)
        px, v1 = self.project(uv + [1, 0])
        py, v2 = self.project(uv + [0, 1])
        m = np.maximum(np.linalg.norm(px - p0, axis=1), np.linalg.norm(py - p0, axis=1))
        return np.where(v0 & v1 & v2, m, np.inf)

    def check_image_size(self, size) -> None:
        if tuple(size) != tuple(self.image_size):
            raise ValueError(
                f"{self.camera_id}: калибровка сделана для кадра {self.image_size}, "
                f"а пришли пиксели кадра {tuple(size)} — пересчитай координаты или калибровку")

    # --- json
    @classmethod
    def load(cls, path) -> "Camera":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        H = np.asarray(d["H"], dtype=np.float64) if d.get("H") else None
        return cls(d["camera_id"], tuple(d["image_size"]), H,
                   Fisheye.from_dict(d.get("distortion")), d)

    def to_dict(self) -> dict:
        d = dict(self.raw)
        d.update({"camera_id": self.camera_id, "image_size": list(self.image_size),
                  "H": None if self.H is None else self.H.tolist(),
                  "distortion": None if self.distortion is None else self.distortion.to_dict()})
        return d


def normalize_H_sign(H: np.ndarray, ideal_pts) -> np.ndarray:
    """Масштаб H произвольный; выбираем знак так, чтобы w > 0 для точек пола."""
    xy = np.asarray(ideal_pts, dtype=np.float64).reshape(-1, 2)
    w = np.column_stack([xy, np.ones(len(xy))]) @ H[2]
    H = H / np.linalg.norm(H)
    return H if np.median(w) > 0 else -H
