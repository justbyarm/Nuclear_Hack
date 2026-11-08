"""S1b: грубая проекция точки ног на план и фильтр «человек стоит на полу».

    python -m mct.geometry.floor_check --detections outputs/run/detections.parquet \
        --calib calib --out outputs/run/floor_check.parquet

Выход (контракт): det_id, foot_u, foot_v, plan_x_raw, plan_y_raw, on_floor, in_fov,
плюс foot_src (mask | bbox) и cut_bottom.

on_floor = точка ног внутри полигона пола плана с допуском tol (люди у стен не теряются).
Отражение в зеркале, человек на экране или постере дают точку «в стене» или «за стеной».
in_fov — мягкий флаг: точка в зоне обзора этой камеры (по калибровке). Для отсева не используем.
Камера без калибровки: on_floor = True, plan_* = NaN (ничего не выбрасываем).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import polygons as P
from .camera import Camera
from .foot import foot_point
from .plan import Plan

OUT_COLUMNS = ["det_id", "foot_u", "foot_v", "plan_x_raw", "plan_y_raw", "on_floor", "in_fov",
               "foot_src", "cut_bottom"]


def load_cameras(calib_dir) -> dict:
    cams = {}
    for p in sorted(Path(calib_dir).glob("*.json")):
        if p.name in ("plan.json", "graph.json"):
            continue
        try:
            cam = Camera.load(p)
        except (KeyError, ValueError):
            continue
        if cam.H is not None or cam.pose() is not None:
            cams[cam.camera_id] = cam
    return cams


def run(dets: pd.DataFrame, cams: dict, plan: Plan, tol_m: float = 0.4,
        bottom_rows: int = 4, cut_margin_px: int = 3, fov_tol_m: float = 0.5) -> pd.DataFrame:
    dets = dets.sort_values("det_id", kind="stable").reset_index(drop=True)
    n = len(dets)
    out = pd.DataFrame({
        "det_id": dets["det_id"].to_numpy(),
        "foot_u": np.full(n, np.nan), "foot_v": np.full(n, np.nan),
        "plan_x_raw": np.full(n, np.nan), "plan_y_raw": np.full(n, np.nan),
        "on_floor": np.ones(n, bool), "in_fov": np.ones(n, bool),
        "foot_src": np.full(n, "bbox", dtype=object), "cut_bottom": np.zeros(n, bool),
    })
    has_mask = "mask_rle" in dets.columns
    for cam_id, idx in dets.groupby("camera_id", sort=True).indices.items():
        cam = cams.get(cam_id)
        size = cam.image_size if cam is not None else None
        sub = dets.iloc[idx]
        feet = []
        for row in sub.itertuples(index=False):
            m = getattr(row, "mask_rle") if has_mask else None
            sz = size or _size_from_rle(m) or (int(row.x2) + 1, int(row.y2) + 1)
            feet.append(foot_point(m, (row.x1, row.y1, row.x2, row.y2), sz, bottom_rows, cut_margin_px))
        u = np.array([f[0] for f in feet]); v = np.array([f[1] for f in feet])
        out.loc[idx, "foot_u"] = u
        out.loc[idx, "foot_v"] = v
        out.loc[idx, "foot_src"] = [f[2] for f in feet]
        out.loc[idx, "cut_bottom"] = [f[3] for f in feet]
        if cam is None:
            continue
        xy, ok = cam.project(np.column_stack([u, v]))
        out.loc[idx, "plan_x_raw"] = xy[:, 0]
        out.loc[idx, "plan_y_raw"] = xy[:, 1]
        floor = plan.on_floor(xy, tol_m) & ok
        # Обрезанный снизу человек: низ маски — не ноги, проекция ненадёжна. Не выбрасываем.
        floor |= out.loc[idx, "cut_bottom"].to_numpy(bool)
        out.loc[idx, "on_floor"] = floor
        fov = cam.raw.get("fov_plan")
        if fov and len(fov) >= 3:
            out.loc[idx, "in_fov"] = ok & P.contains_chunked(fov, np.nan_to_num(xy, nan=-1e6), fov_tol_m)
        else:
            out.loc[idx, "in_fov"] = ok
    return out[OUT_COLUMNS]


def _size_from_rle(m):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return None
    try:
        from .rle import parse
        H, W, _ = parse(m)
        return (W, H)
    except Exception:  # noqa: BLE001
        return None


def summary(fc: pd.DataFrame, dets: pd.DataFrame) -> pd.DataFrame:
    d = fc.merge(dets[["det_id", "camera_id"]], on="det_id")
    g = d.groupby("camera_id")
    return pd.DataFrame({"n": g.size(), "off_floor_pct": 100 * (1 - g["on_floor"].mean()),
                         "out_of_fov_pct": 100 * (1 - g["in_fov"].mean()),
                         "bbox_fallback_pct": 100 * g["foot_src"].apply(lambda s: (s == "bbox").mean()),
                         "cut_bottom_pct": 100 * g["cut_bottom"].mean()}).round(2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detections", required=True)
    ap.add_argument("--calib", default="calib")
    ap.add_argument("--plan", default=None, help="по умолчанию <calib>/plan.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=0.4)
    a = ap.parse_args(argv)
    dets = pd.read_parquet(a.detections)
    plan = Plan.load(a.plan or Path(a.calib) / "plan.json")
    cams = load_cameras(a.calib)
    print(f"камеры с калибровкой: {sorted(cams)}")
    fc = run(dets, cams, plan, a.tol)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(a.out, index=False)
    print(summary(fc, dets).to_string())
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
