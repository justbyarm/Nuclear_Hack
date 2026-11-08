"""Путь глобального трека на плане: наблюдаемые участки и предполагаемый путь вне обзора.

Логика без UI (её использует app/pages/3_map.py и карточка трека У1):
  1. Треклеты трека интерполируются на сетку времени (0.1 с).
  2. В холле человека видят 2–3 камеры сразу: позиции сливаются средним с весами 1/σ²,
     иначе на карте три параллельные линии. Цвет участка — камера с наименьшим σ.
  3. Промежуток без наблюдений дольше gap_s — «предполагаемый путь»: кратчайший путь
     по опорным точкам коридоров (plan.nav_nodes), чтобы пунктир не шёл сквозь стены.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


@dataclass
class TrackPath:
    global_id: object
    observed: list = field(default_factory=list)   # [{"camera", "t", "x", "y", "sigma"}]
    gaps: list = field(default_factory=list)       # [{"t0", "t1", "x", "y", "routed"}]
    t_start: float = np.nan
    t_end: float = np.nan
    cameras: list = field(default_factory=list)

    @property
    def start(self):
        return (self.observed[0]["x"][0], self.observed[0]["y"][0]) if self.observed else None

    @property
    def end(self):
        return (self.observed[-1]["x"][-1], self.observed[-1]["y"][-1]) if self.observed else None


def fuse(traj: pd.DataFrame, cam_of: dict, dt: float = 0.1, max_interp_s: float = 0.6):
    """traj — строки trajectories.parquet одного трека. -> DataFrame t, x, y, sigma, camera."""
    if traj.empty:
        return pd.DataFrame(columns=["t", "x", "y", "sigma", "camera"])
    t0, t1 = traj["t"].min(), traj["t"].max()
    grid = np.round(np.arange(t0, t1 + dt / 2, dt), 3)
    W = np.zeros(len(grid)); SX = np.zeros(len(grid)); SY = np.zeros(len(grid))
    best_s = np.full(len(grid), np.inf); best_c = np.full(len(grid), None, dtype=object)
    for tid, tr in traj.groupby("tracklet_id", sort=True):
        tr = tr.sort_values("t")
        t = tr["t"].to_numpy(float)
        a, b = np.searchsorted(grid, t[0] - 1e-9), np.searchsorted(grid, t[-1] + 1e-9)
        g = grid[a:b]
        if len(g) == 0:
            continue
        # Внутри треклета не интерполируем через дыры длиннее max_interp_s.
        k = np.clip(np.searchsorted(t, g), 1, len(t) - 1)
        near = np.minimum(np.abs(t[k] - g), np.abs(t[k - 1] - g))
        ok = (near <= max_interp_s / 2) | (len(t) == 1)
        x = np.interp(g, t, tr["x"]); y = np.interp(g, t, tr["y"]); s = np.interp(g, t, tr["sigma"])
        w = np.where(ok, 1.0 / np.maximum(s, 0.03) ** 2, 0.0)
        W[a:b] += w; SX[a:b] += w * x; SY[a:b] += w * y
        better = ok & (s < best_s[a:b])
        idx = np.arange(a, b)[better]
        best_s[idx] = s[better]; best_c[idx] = cam_of.get(tid)
    seen = W > 0
    out = pd.DataFrame({"t": grid[seen], "x": SX[seen] / W[seen], "y": SY[seen] / W[seen],
                        "sigma": 1.0 / np.sqrt(W[seen]), "camera": best_c[seen]})
    return out.reset_index(drop=True)


class Router:
    """Кратчайший путь по полу через опорные точки коридоров."""

    def __init__(self, plan, link_m: float = 3.3):
        self.plan = plan
        self.nodes = np.asarray(plan.nav_nodes, float).reshape(-1, 2)
        self.link = link_m
        n = len(self.nodes)
        rows, cols, w = [], [], []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(self.nodes[i] - self.nodes[j])
                if d <= link_m and self._clear(self.nodes[i], self.nodes[j]):
                    rows += [i, j]; cols += [j, i]; w += [d, d]
        self.G = csr_matrix((w, (rows, cols)), shape=(n, n))

    def _clear(self, a, b, tol=0.3):
        seg = a + (b - a) * np.linspace(0, 1, max(3, int(np.linalg.norm(b - a) / 0.2)))[:, None]
        return bool(self.plan.on_floor(seg, tol).all())

    def _attach(self, p):
        if len(self.nodes) == 0:
            return []
        d = np.linalg.norm(self.nodes - p, axis=1)
        order = np.argsort(d)[:6]
        return [(int(i), float(d[i])) for i in order if d[i] <= 2 * self.link and self._clear(p, self.nodes[i])]

    def route(self, a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if self._clear(a, b, 0.2) or len(self.nodes) == 0:
            return np.array([a, b]), False
        A, B = self._attach(a), self._attach(b)
        if not A or not B:
            return np.array([a, b]), False
        best = None
        for i, da in A:
            dist, pred = dijkstra(self.G, indices=i, return_predecessors=True)
            for j, db in B:
                tot = da + dist[j] + db
                if np.isfinite(tot) and (best is None or tot < best[0]):
                    path, k = [j], j
                    while k != i:
                        k = pred[k]; path.append(k)
                    best = (tot, path[::-1])
        if best is None:
            return np.array([a, b]), False
        return np.vstack([a, self.nodes[best[1]], b]), True


def build(global_id, traj: pd.DataFrame, tracklets: pd.DataFrame, router: Router | None = None,
          gap_s: float = 0.5, dt: float = 0.1) -> TrackPath:
    cam_of = dict(zip(tracklets["tracklet_id"], tracklets["camera_id"]))
    f = fuse(traj, cam_of, dt)
    tp = TrackPath(global_id)
    if f.empty:
        return tp
    tp.t_start, tp.t_end = float(f["t"].iloc[0]), float(f["t"].iloc[-1])
    tp.cameras = sorted({c for c in f["camera"] if c is not None})
    brk = np.flatnonzero(np.diff(f["t"].to_numpy()) > gap_s) + 1
    pieces = np.split(np.arange(len(f)), brk)
    for n, idx in enumerate(pieces):
        part = f.iloc[idx]
        # Внутри наблюдаемого куска — отрезки одного цвета (камеры).
        cams = part["camera"].to_numpy()
        cut = np.flatnonzero(cams[1:] != cams[:-1]) + 1
        for sub in np.split(np.arange(len(part)), cut):
            # Перекрываем стык одной точкой, чтобы линия не рвалась при смене цвета.
            s = np.r_[sub, sub[-1] + 1] if sub[-1] + 1 < len(part) else sub
            q = part.iloc[s]
            tp.observed.append({"camera": cams[sub[0]], "t": q["t"].to_numpy(), "x": q["x"].to_numpy(),
                                "y": q["y"].to_numpy(), "sigma": q["sigma"].to_numpy()})
        if n + 1 < len(pieces):
            a = part.iloc[-1]; b = f.iloc[pieces[n + 1][0]]
            pts, routed = router.route((a.x, a.y), (b.x, b.y)) if router else (np.array([[a.x, a.y], [b.x, b.y]]), False)
            tp.gaps.append({"t0": float(a.t), "t1": float(b.t), "x": pts[:, 0], "y": pts[:, 1], "routed": routed})
    return tp


def position_at(tp: TrackPath, t: float):
    """Положение на момент t: (x, y, observed). None — трек в этот момент не существует."""
    for s in tp.observed:
        if s["t"][0] <= t <= s["t"][-1]:
            return float(np.interp(t, s["t"], s["x"])), float(np.interp(t, s["t"], s["y"])), True
    for g in tp.gaps:
        if g["t0"] <= t <= g["t1"]:
            L = np.r_[0, np.cumsum(np.hypot(np.diff(g["x"]), np.diff(g["y"])))]
            u = (t - g["t0"]) / max(g["t1"] - g["t0"], 1e-9) * L[-1]
            return float(np.interp(u, L, g["x"])), float(np.interp(u, L, g["y"])), False
    return None
