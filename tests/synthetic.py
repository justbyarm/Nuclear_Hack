"""Синтетическая fisheye-камера над полом — для тестов без реальных кадров."""
from __future__ import annotations

import numpy as np

from mct.geometry.camera import Fisheye


class SynthCam:
    def __init__(self, pos=(5.0, 0.0, 2.8), pitch_deg=45.0, yaw_deg=0.0,
                 model=Fisheye(650.0, 1300.0, 950.0, (-0.03, 0.004, 0.0, 0.0)),
                 size=(2560, 1920)):
        self.C = np.asarray(pos, float)
        self.model = model
        self.size = size
        p, y = np.deg2rad(pitch_deg), np.deg2rad(yaw_deg)
        fwd = np.array([np.sin(y) * np.cos(p), np.cos(y) * np.cos(p), -np.sin(p)])
        right = np.array([np.cos(y), -np.sin(y), 0.0])
        down = np.cross(fwd, right)
        self.R = np.stack([right, down, fwd])          # мир -> камера

    def world_to_pixels(self, Q):
        Q = np.asarray(Q, float).reshape(-1, 3)
        c = (Q - self.C) @ self.R.T
        b = c / np.linalg.norm(c, axis=1, keepdims=True)
        uv = self.model.bearings_to_pixels(b)
        W, H = self.size
        th = np.degrees(np.arccos(np.clip(b[:, 2], -1, 1)))
        vis = (th < 85) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        return uv, vis

    def floor_to_pixels(self, XY):
        XY = np.asarray(XY, float).reshape(-1, 2)
        return self.world_to_pixels(np.column_stack([XY, np.zeros(len(XY))]))

    def floor_lines(self, xs=(2, 3.2, 4.4, 5.6, 6.8, 8), ys=(1.5, 3, 4.5, 6, 7.5), n=15):
        """Швы плитки: прямые на полу -> ломаные на кадре (только видимые точки)."""
        lines = []
        for x in xs:
            uv, vis = self.floor_to_pixels(np.column_stack([np.full(n, x), np.linspace(0.5, 9, n)]))
            if vis.sum() >= 5:
                lines.append(uv[vis])
        for y in ys:
            uv, vis = self.floor_to_pixels(np.column_stack([np.linspace(0, 10, n), np.full(n, y)]))
            if vis.sum() >= 5:
                lines.append(uv[vis])
        return lines
