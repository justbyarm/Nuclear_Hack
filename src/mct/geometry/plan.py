"""План офиса: масштаб, пол, выходы, препятствия. Всё хранится в метрах.

Начало координат — левый верхний угол картинки чистого плана, ось y вниз.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import polygons as P


def scale_from_segments(segments_px, lengths_m) -> dict:
    """Масштаб по отрезкам известной длины (например, модули «1 200» мм).

    segments_px: [[(x1, y1), (x2, y2)], ...] в пикселях плана; lengths_m: длины в метрах.
    """
    segs = np.asarray(segments_px, dtype=np.float64).reshape(-1, 2, 2)
    L = np.asarray(lengths_m, dtype=np.float64).reshape(-1)
    px = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
    m_per_px = L / px
    return {"m_per_px": float(np.median(m_per_px)),
            "spread_pct": float(100 * (m_per_px.max() - m_per_px.min()) / np.median(m_per_px)),
            "n": int(len(L)), "each": m_per_px.tolist()}


@dataclass
class Plan:
    m_per_px: float
    image_clean: str = ""
    size_m: tuple = (0.0, 0.0)
    floor: list = field(default_factory=list)          # [ [[x, y], ...], ... ] метры
    exits: list = field(default_factory=list)          # [{"id", "type", "polygon"}]
    occluders: list = field(default_factory=list)      # [{"id", "polygon"}]
    nav_nodes: list = field(default_factory=list)      # [[x, y], ...]
    raw: dict = field(default_factory=dict)

    # --- пиксели плана <-> метры
    def px_to_m(self, pts):
        return np.asarray(pts, dtype=np.float64) * self.m_per_px

    def m_to_px(self, pts):
        return np.asarray(pts, dtype=np.float64) / self.m_per_px

    # --- запросы
    def on_floor(self, pts_m, tol_m: float = 0.0) -> np.ndarray:
        pts = np.asarray(pts_m, dtype=np.float64).reshape(-1, 2)
        out = np.zeros(len(pts), dtype=bool)
        fin = np.isfinite(pts).all(1)
        for poly in self.floor:
            out[fin] |= P.contains_chunked(poly, pts[fin], tol_m)
        return out

    def exit_at(self, pts_m) -> list:
        pts = np.asarray(pts_m, dtype=np.float64).reshape(-1, 2)
        res = [None] * len(pts)
        for e in self.exits:
            hit = P.contains(e["polygon"], pts)
            for i in np.where(hit)[0]:
                res[i] = res[i] or e["id"]
        return res

    # --- json
    @classmethod
    def load(cls, path) -> "Plan":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["m_per_px"], d.get("image_clean", ""), tuple(d.get("size_m", (0, 0))),
                   d.get("floor", []), d.get("exits", []), d.get("occluders", []),
                   d.get("nav_nodes", []), d)

    def to_dict(self) -> dict:
        d = dict(self.raw)
        d.update({"m_per_px": self.m_per_px, "image_clean": self.image_clean,
                  "size_m": list(self.size_m), "floor": self.floor, "exits": self.exits,
                  "occluders": self.occluders, "nav_nodes": self.nav_nodes})
        return d

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
