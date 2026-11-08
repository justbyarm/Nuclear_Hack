"""Карта трека для интерфейса (Plotly). Используют app/pages/3_map.py и карточка трека У1:

    from mct.geometry.map_view import MapData, track_figure
    md = MapData.load("outputs/run", "calib")
    fig = track_figure(md, global_id=17)           # st.plotly_chart(fig, use_container_width=True)

Сплошная линия — наблюдаемый путь (цвет = камера), пунктир — предполагаемый путь вне обзора.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from .plan import Plan
from .track_path import Router, build, position_at

CAM_COLORS = {"cam1": "#1f77b4", "cam2": "#2ca02c", "cam3": "#d62728", "cam4": "#9467bd", "cam5": "#ff7f0e"}


@dataclass
class MapData:
    plan: Plan
    plan_image: Path | None
    fovs: dict
    traj: pd.DataFrame
    tracklets: pd.DataFrame
    assign: pd.DataFrame          # tracklet_id, global_id
    tracks: pd.DataFrame | None
    source: str

    @classmethod
    def load(cls, run_dir, calib_dir="calib", repo_root=None):
        run_dir, calib_dir = Path(run_dir), Path(calib_dir)
        plan = Plan.load(calib_dir / "plan.json")
        root = Path(repo_root) if repo_root else calib_dir.parent
        img = None
        for cand in (plan.image_clean, "data/plan_clean.png"):
            if cand and (root / cand).exists():
                img = root / cand
                break
        fovs = {}
        for p in sorted(calib_dir.glob("*.json")):
            if p.name in ("plan.json", "graph.json"):
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("fov_plan"):
                fovs[d.get("camera_id", p.stem)] = d["fov_plan"]
        if not fovs and (calib_dir / "fake_fovs.json").exists():
            fovs = json.loads((calib_dir / "fake_fovs.json").read_text())
        traj = pd.read_parquet(run_dir / "trajectories.parquet")
        trk = pd.read_parquet(run_dir / "tracklets.parquet")
        tracks = pd.read_csv(run_dir / "tracks.csv") if (run_dir / "tracks.csv").exists() else None
        if (run_dir / "assignments.parquet").exists():
            assign, source = pd.read_parquet(run_dir / "assignments.parquet"), "assignments"
        elif (run_dir / "gt_tracklets.csv").exists():          # фейки: эталон вместо склейки
            gt = pd.read_csv(run_dir / "gt_tracklets.csv")
            assign, source = gt[["tracklet_id"]].assign(global_id=gt["gt_transit"]), "эталон фейков"
        else:
            raise FileNotFoundError(f"нет assignments.parquet в {run_dir}")
        return cls(plan, img, fovs, traj, trk, assign[["tracklet_id", "global_id"]], tracks, source)

    @cached_property
    def router(self):
        return Router(self.plan)

    def global_ids(self):
        return sorted(self.assign["global_id"].unique().tolist())

    def path(self, global_id):
        tids = self.assign.loc[self.assign["global_id"] == global_id, "tracklet_id"]
        return build(global_id, self.traj[self.traj["tracklet_id"].isin(tids)], self.tracklets, self.router)


def _hex_rgba(h, a):
    h = h.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def track_figure(md: MapData, global_id, t_now: float | None = None, show_fov: bool = True, height: int = 700):
    import plotly.graph_objects as go
    tp = md.path(global_id)
    W, H = md.plan.size_m
    fig = go.Figure()
    if md.plan_image is not None:
        from PIL import Image
        fig.add_layout_image(source=Image.open(md.plan_image), x=0, y=0, sizex=W, sizey=H, xref="x", yref="y",
                             sizing="stretch", layer="below", opacity=0.9)
    if show_fov:
        for cam, poly in md.fovs.items():
            p = np.asarray(poly, float)
            c = CAM_COLORS.get(cam, "#888888")
            fig.add_trace(go.Scatter(x=np.r_[p[:, 0], p[0, 0]], y=np.r_[p[:, 1], p[0, 1]], fill="toself",
                                     fillcolor=_hex_rgba(c, 0.10), line=dict(color=_hex_rgba(c, 0.5), width=1),
                                     name=f"зона {cam}", hoverinfo="name", legendgroup="fov"))
    shown = set()
    for s in tp.observed:
        c = CAM_COLORS.get(s["camera"], "#333333")
        fig.add_trace(go.Scatter(
            x=s["x"], y=s["y"], mode="lines", line=dict(color=c, width=4),
            name=f"виден: {s['camera']}", legendgroup=s["camera"], showlegend=s["camera"] not in shown,
            customdata=np.column_stack([s["t"], s["sigma"]]),
            hovertemplate=f"{s['camera']}<br>t=%{{customdata[0]:.1f}} с<br>σ=%{{customdata[1]:.2f}} м<extra></extra>"))
        shown.add(s["camera"])
    for i, g in enumerate(tp.gaps):
        fig.add_trace(go.Scatter(x=g["x"], y=g["y"], mode="lines", line=dict(color="#444444", width=3, dash="dash"),
                                 name="вне обзора (предполагаемый путь)", legendgroup="gap", showlegend=i == 0,
                                 hovertemplate=f"вне обзора {g['t0']:.1f}–{g['t1']:.1f} с, "
                                               f"{g['t1'] - g['t0']:.1f} с<extra></extra>"))
    if tp.observed:
        (sx, sy), (ex, ey) = tp.start, tp.end
        fig.add_trace(go.Scatter(x=[sx], y=[sy], mode="markers+text", marker=dict(size=13, color="#00a651", symbol="circle"),
                                 text=[f"старт {tp.t_start:.1f} с"], textposition="middle left", name="начало"))
        fig.add_trace(go.Scatter(x=[ex], y=[ey], mode="markers+text", marker=dict(size=13, color="#e31a1c", symbol="square"),
                                 text=[f"конец {tp.t_end:.1f} с"], textposition="middle right", name="конец"))
    if t_now is not None:
        p = position_at(tp, t_now)
        if p is not None:
            fig.add_trace(go.Scatter(x=[p[0]], y=[p[1]], mode="markers",
                                     marker=dict(size=18, color="#000000" if p[2] else "#ffffff",
                                                 line=dict(color="#000000", width=3)),
                                     name="сейчас" + ("" if p[2] else " (вне обзора)")))
    # Обзор на путь трека с запасом, масштаб 1:1, ось y вниз как у плана.
    if tp.observed:
        xs = np.concatenate([s["x"] for s in tp.observed] + [g["x"] for g in tp.gaps])
        ys = np.concatenate([s["y"] for s in tp.observed] + [g["y"] for g in tp.gaps])
        pad = 4.0
        x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad)
        y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad)
        # Кадр с пропорциями ~4:3, чтобы план не сжимался в полоску.
        w, h = x1 - x0, y1 - y0
        if w < 1.2 * h:
            cx = (x0 + x1) / 2; x0, x1 = max(0, cx - 0.6 * h), min(W, cx + 0.6 * h)
    else:
        x0, x1, y0, y1 = 0, W, 0, H
    fig.update_xaxes(range=[x0, x1], showgrid=False, zeroline=False, title="м", constrain="domain")
    fig.update_yaxes(range=[y1, y0], scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False, constrain="domain")
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.05),
                      title=f"Трек {global_id}: {tp.t_start:.1f}–{tp.t_end:.1f} с, камеры {', '.join(tp.cameras)}")
    return fig, tp
