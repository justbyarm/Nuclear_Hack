"""S5: траектории треклетов на плане.

    python -m mct.geometry.trajectories --run outputs/run --calib calib

Читает из --run: detections.parquet (У1), tracklet_dets.parquet (У4), floor_check.parquet (S1b, если есть).
Пишет trajectories.parquet:
    det_id, tracklet_id, t, x, y, sigma, zone, uncertain        — контракт
    vx, vy, cams_cover (JSON-список), x_raw, y_raw             — дополнительно

sigma измерения = sigma0 + k · (метров на пиксель в точке ног), × mult при uncertain.
uncertain: точка ног по рамке (нет маски), человек обрезан краем кадра, выброс фильтра.
Камеры без калибровки: строки не пишутся (склейка видит треклет без позиции).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .camera import Camera
from .floor_check import load_cameras
from .feet import recover
from .foot import foot_point
from .rle import top_info
from .plan import Plan
from .polygons import nearest_on_polys
from .smoothing import kalman_rts, rolling_median
from .zones import ZoneMap

COLUMNS = ["det_id", "tracklet_id", "t", "x", "y", "sigma", "zone", "uncertain",
           "vx", "vy", "cams_cover", "x_raw", "y_raw"]

DEFAULTS = dict(sigma0_m=0.15, sigma_k_px=3.0, uncertain_mult=3.0, median_win=5,
                kalman_q=0.5, outlier_gate=3.5, v_max=2.5,
                floor_tol_m=0.5,      # точка дальше от пола — не участвует в сглаживании
                max_range_m=35.0,     # грубый предел дальности (защита от мусора)
                max_mpp=0.10,         # больше 10 см на пиксель — у горизонта, не верим
                max_interp_s=0.5,     # (не используется: точки больше не выбрасываются)
                fallback_sigma_m=2.0, # треклет без единой надёжной точки: σ
                min_valid_frac=0.3,   # меньше годных точек в треклете — весь треклет uncertain
                jump_speed=8.0,       # последняя страховка на выходе: шаг быстрее — кусок удерживается на месте (uncertain)
                short_frac=0.65,      # рамка короче 65% медианы по ±0.5 с — ноги не видны (закрыты)
                short_win_s=0.5,
                edge_px=3,            # рамка у края кадра — ноги не видны
                low_conf=0.35, low_conf_frac=0.75,  # неуверенная детекция + рамка < 75% медианы — пропуск
                spike_speed=3.0,      # скачок быстрее 3 м/с — точку не берём, восстанавливаем по соседям
                spike_min_m=0.6)


def _feet(dets: pd.DataFrame, fc: pd.DataFrame | None, cams: dict) -> pd.DataFrame:
    """foot_u, foot_v, cut_bottom, foot_src для каждой детекции (из S1b или заново)."""
    if fc is not None and {"foot_u", "foot_v"} <= set(fc.columns):
        cols = ["det_id", "foot_u", "foot_v"] + [c for c in ("cut_bottom", "foot_src") if c in fc.columns]
        d = dets.merge(fc[cols], on="det_id", how="left")
    else:
        d = dets.copy()
        d["foot_u"] = np.nan
    miss = d["foot_u"].isna().to_numpy()
    if miss.any():
        has_mask = "mask_rle" in d.columns
        for i in np.flatnonzero(miss):
            r = d.iloc[i]
            cam = cams.get(r["camera_id"])
            size = cam.image_size if cam is not None else (int(r["x2"]) + 1, int(r["y2"]) + 1)
            u, v, src, cut = foot_point(r["mask_rle"] if has_mask else None, (r.x1, r.y1, r.x2, r.y2), size)
            d.iat[i, d.columns.get_loc("foot_u")] = u
            if "foot_v" not in d.columns:
                d["foot_v"] = np.nan
            d.iat[i, d.columns.get_loc("foot_v")] = v
            for c, val in (("cut_bottom", cut), ("foot_src", src)):
                if c not in d.columns:
                    d[c] = None
                d.iat[i, d.columns.get_loc(c)] = val
    for c, dflt in (("cut_bottom", False), ("foot_src", "mask")):
        if c not in d.columns:
            d[c] = dflt
    d["cut_bottom"] = d["cut_bottom"].astype(object).where(d["cut_bottom"].notna(), False).astype(bool)
    d["foot_src"] = d["foot_src"].fillna("mask")
    # Верх маски — для восстановления положения, когда ноги закрыты.
    tops = np.full((len(d), 2), np.nan)
    if "mask_rle" in d.columns:
        for i, m in enumerate(d["mask_rle"].to_numpy()):
            if isinstance(m, (str, dict)) and m != "":
                try:
                    ti = top_info(m)
                except (ValueError, KeyError, TypeError, IndexError):
                    ti = None
                if ti is not None:
                    tops[i] = (ti[1], ti[0])
    # Без маски верх рамки не годится: макушка не на её середине. Такие точки не восстанавливаем.
    d["top_u"], d["top_v"] = tops[:, 0], tops[:, 1]
    return d


def run(dets: pd.DataFrame, tdets: pd.DataFrame, cams: dict, plan: Plan, fc: pd.DataFrame | None = None,
        zonemap: ZoneMap | None = None, **kw) -> pd.DataFrame:
    p = {**DEFAULTS, **kw}
    d = dets.merge(tdets[["det_id", "tracklet_id"]], on="det_id", how="inner")
    d = _feet(d, fc, cams)
    d = d[d["camera_id"].isin(list(cams))].sort_values(["tracklet_id", "t", "det_id"], kind="stable")
    rows = []
    split_tracklets = []
    no_support = []
    for cam_id, g in d.groupby("camera_id", sort=True):
        cam: Camera = cams[cam_id]
        uv = g[["foot_u", "foot_v"]].to_numpy(float)
        # Ноги не видны: рамка резко короче соседних (человека заслонили) или у края кадра.
        Wf, Hf = cam.image_size
        hbox = (g["y2"] - g["y1"]).to_numpy(float)
        short = np.zeros(len(g), bool)
        for _, ix in g.groupby("tracklet_id", sort=False).indices.items():
            tt = g["t"].to_numpy(float)[ix]; hh = hbox[ix]
            lo = np.searchsorted(tt, tt - p["short_win_s"]); hi = np.searchsorted(tt, tt + p["short_win_s"], side="right")
            med = np.array([np.median(hh[a:b]) for a, b in zip(lo, hi)])
            short[ix] = hh < p["short_frac"] * med
            if "conf" in g.columns:
                cf = g["conf"].to_numpy(float)[ix]
                short[ix] |= (cf < p["low_conf"]) & (hh < p["low_conf_frac"] * med)
        e = p["edge_px"]
        edge = (g["x1"].to_numpy() <= e) | (g["x2"].to_numpy() >= Wf - 1 - e) | (g["y2"].to_numpy() >= Hf - 1 - e)
        hidden = g["cut_bottom"].to_numpy(bool) | short | edge
        xy, ok = cam.project(uv)
        mpp = cam.meters_per_pixel(uv)
        unc = hidden | (g["foot_src"].to_numpy() == "bbox") | ~ok
        r = p["sigma0_m"] + p["sigma_k_px"] * np.where(np.isfinite(mpp), mpp, 0.05)
        r = np.where(unc, r * p["uncertain_mult"], r)
        if cam.pose() is not None and p.get("recover_feet", True):
            top = g[["top_u", "top_v"]].to_numpy(float)
            has_top = np.isfinite(top).all(1)
            rec = recover(cam, uv, np.where(has_top[:, None], top, uv), hidden,
                          plan.raw.get("objects", []) if plan.raw else [])
            rec["src"] = np.where(has_top, rec["src"], "foot")
            rec["extra_sigma"] = np.where(has_top, rec["extra_sigma"], 0.0)
            fix = rec["src"] != "foot"
            head = rec["src"] == "head"
            xy = np.where(head[:, None], rec["xy"], xy)
            ok = ok | (head & np.isfinite(rec["xy"]).all(1))
            # Восстановленная по макушке точка надёжнее обрезанных ног, но всё равно uncertain.
            r = np.where(head, p["sigma0_m"] + rec["extra_sigma"] + p["sigma_k_px"] * np.nan_to_num(mpp, nan=0.05, posinf=0.5), r)
            r = r + np.where(rec["src"] == "seated", rec["extra_sigma"], 0)
            unc = unc | fix
        # Годная точка: проекция есть, на полу (с допуском) и не у горизонта.
        pose = cam.pose()
        dist = np.linalg.norm(xy - pose[1][:2], axis=1) if pose is not None else np.zeros(len(xy))
        good = ok & np.isfinite(xy).all(1) & plan.on_floor(np.nan_to_num(xy, nan=-1e6), p["floor_tol_m"]) \
            & (dist < p["max_range_m"]) & (np.nan_to_num(mpp, nan=1e9, posinf=1e9) < p["max_mpp"])
        # Ноги не видны и восстановить по макушке нечем — точку ног не берём.
        head_ok = np.zeros(len(g), bool)
        if cam.pose() is not None and p.get("recover_feet", True):
            head_ok = head
        good &= ~hidden | head_ok
        gg = g.assign(x_raw=xy[:, 0], y_raw=xy[:, 1], r=r, unc=unc, good=good)
        for tid, tr in gg.groupby("tracklet_id", sort=True):
            t = tr["t"].to_numpy(float)
            gd = tr["good"].to_numpy(bool)
            zr = tr[["x_raw", "y_raw"]].to_numpy(float)
            # Скачки (> spike_speed): точку не берём, восстанавливаем по соседям. Треклет не режем.
            rr = tr["r"].to_numpy(float)
            for _ in range(max(1, int(0.3 * len(t)))):
                ig = np.flatnonzero(gd)
                if len(ig) < 3:
                    break
                st = np.hypot(*np.diff(zr[ig], axis=0).T)
                sp = st / np.maximum(np.diff(t[ig]), 1e-3)
                lim = np.maximum(p["spike_min_m"], 1.5 * (rr[ig][:-1] + rr[ig][1:]))
                viol = np.flatnonzero((sp > p["spike_speed"]) & (st > lim))
                if not len(viol):
                    break
                med = rolling_median(zr[ig], 7)
                dev = np.hypot(*(zr[ig] - med).T)
                k = viol[np.argmax(np.maximum(dev[viol], dev[viol + 1]))]
                worst = ig[k] if dev[k] >= dev[k + 1] else ig[k + 1]
                gd[worst] = False
            fin = np.isfinite(zr).all(1)
            if gd.sum() < 2:
                # Опоры нет (лестница, отражение, дальний план): точку всё равно выдаём, но неуверенную —
                # в зоне обзора своей камеры, на постоянном месте, с большой σ.
                fov = cam.raw.get("fov_plan")
                if gd.sum() == 1:
                    c0 = zr[gd][0]
                elif fin.any():
                    c0 = np.median(zr[fin], axis=0)
                elif fov:
                    c0 = np.mean(np.asarray(fov, float), axis=0)
                else:
                    c0 = np.asarray(plan.size_m, float) / 2
                if fov:
                    c0 = nearest_on_polys([fov], c0[None])[0]
                if plan.floor:
                    c0 = nearest_on_polys(plan.floor, c0[None])[0]
                xs = np.tile(c0, (len(t), 1)); vs = np.zeros((len(t), 2))
                sig = np.full(len(t), p["fallback_sigma_m"]); unc_t = np.ones(len(t), bool)
                rows.append(pd.DataFrame({
                    "det_id": tr["det_id"].to_numpy(), "tracklet_id": tid, "t": t,
                    "x": xs[:, 0], "y": xs[:, 1], "sigma": sig, "uncertain": unc_t,
                    "vx": vs[:, 0], "vy": vs[:, 1], "x_raw": zr[:, 0], "y_raw": zr[:, 1]}))
                no_support.append(int(tid) if np.issubdtype(type(tid), np.integer) else tid)
                continue
            tg = t[gd]
            zg = rolling_median(zr[gd], p["median_win"])
            xg, vg, sg, og = kalman_rts(tg, zg, tr["r"].to_numpy(float)[gd], p["kalman_q"], p["outlier_gate"])
            k = np.clip(np.searchsorted(tg, t), 1, len(tg) - 1)
            gap = np.minimum(np.abs(t - tg[k]), np.abs(t - tg[k - 1]))
            # Ненадёжные точки: положение по соседним надёжным (внутри) или последнее надёжное (по краям).
            xs = np.column_stack([np.interp(t, tg, xg[:, 0]), np.interp(t, tg, xg[:, 1])])
            vs = np.column_stack([np.interp(t, tg, vg[:, 0]), np.interp(t, tg, vg[:, 1])])
            vs[(t < tg[0]) | (t > tg[-1])] = 0.0
            # σ сглаживателя слишком оптимистична (сравнение камер даёт медиану 0.35 м), поэтому
            # не меньше погрешности одного замера.
            sig = np.maximum(np.interp(t, tg, sg), np.interp(t, tg, tr["r"].to_numpy(float)[gd]))
            sig = np.where(gd, sig, sig + 0.3 + np.minimum(gap, 2.0))
            outl = np.zeros(len(t), bool); outl[np.flatnonzero(gd)] = og
            if plan.floor:
                xs = nearest_on_polys(plan.floor, xs)
            unc_t = tr["unc"].to_numpy(bool) | outl | ~gd
            if gd.mean() < p["min_valid_frac"]:
                unc_t[:] = True
            # Страховка: невозможных скачков в выдаче быть не должно. Самый длинный кусок без них
            # оставляем как есть, остальные точки удерживаем на его ближайшем конце (uncertain).
            if len(t) > 1:
                st = np.hypot(*np.diff(xs, axis=0).T)
                bad = (st / np.maximum(np.diff(t), 1e-3) > p["jump_speed"]) & (st > 1.0)
                if bad.any():
                    parts = np.split(np.arange(len(t)), np.flatnonzero(bad) + 1)
                    best = max(parts, key=len)
                    idx = np.arange(len(t))
                    pre, post = idx < best[0], idx > best[-1]
                    xs[pre] = xs[best[0]]; xs[post] = xs[best[-1]]
                    vs[pre | post] = 0.0
                    sig[pre | post] += 1.0
                    unc_t[pre | post] = True
                    if tid not in split_tracklets:
                        split_tracklets.append(int(tid) if np.issubdtype(type(tid), np.integer) else tid)
            rows.append(pd.DataFrame({
                "det_id": tr["det_id"].to_numpy(), "tracklet_id": tid, "t": t,
                "x": xs[:, 0], "y": xs[:, 1], "sigma": np.maximum(sig, 0.05),
                "uncertain": unc_t, "vx": vs[:, 0], "vy": vs[:, 1],
                "x_raw": zr[:, 0], "y_raw": zr[:, 1]}))
    run.split_tracklets = sorted(split_tracklets)
    run.no_support = sorted(no_support)
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    out = pd.concat(rows, ignore_index=True)
    zm = zonemap or ZoneMap(plan, {c: cams[c].raw.get("fov_plan") for c in cams if cams[c].raw.get("fov_plan")})
    xy = out[["x", "y"]].to_numpy()
    out["zone"] = zm.zone(xy)
    out["cams_cover"] = [json.dumps(c) for c in zm.cams_cover(xy)]
    return out[COLUMNS].sort_values(["tracklet_id", "t", "det_id"], kind="stable").reset_index(drop=True)


def _max_step_speed(tr):
    v = 0.0
    for _, g in tr.sort_values("t").groupby("tracklet_id"):
        if len(g) > 1:
            d = np.hypot(np.diff(g["x"]), np.diff(g["y"])) / np.maximum(np.diff(g["t"]), 1e-3)
            v = max(v, float(d.max()))
    return v


def summary(tr: pd.DataFrame, v_max: float = 2.5) -> dict:
    sp = np.hypot(tr["vx"], tr["vy"])
    return {"n_points": int(len(tr)), "n_tracklets": int(tr["tracklet_id"].nunique()),
            "uncertain_pct": round(100 * float(tr["uncertain"].mean()), 2),
            "sigma_median_m": round(float(tr["sigma"].median()), 3),
            "speed_over_vmax_pct": round(100 * float((sp > v_max).mean()), 2),
            "max_step_speed_mps": round(float(_max_step_speed(tr)), 2),
            "x_range_m": [round(float(tr["x"].min()), 2), round(float(tr["x"].max()), 2)],
            "y_range_m": [round(float(tr["y"].min()), 2), round(float(tr["y"].max()), 2)],
            "zones": tr["zone"].str.split(":").str[0].value_counts().to_dict()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="папка outputs/<run>")
    ap.add_argument("--calib", default="calib")
    a = ap.parse_args(argv)
    run_dir = Path(a.run)
    dets = pd.read_parquet(run_dir / "detections.parquet")
    tdets = pd.read_parquet(run_dir / "tracklet_dets.parquet")
    fcp = run_dir / "floor_check.parquet"
    fc = pd.read_parquet(fcp) if fcp.exists() else None
    plan = Plan.load(Path(a.calib) / "plan.json")
    cams = load_cameras(a.calib)
    tr = run(dets, tdets, cams, plan, fc)
    tr.to_parquet(run_dir / "trajectories.parquet", index=False)
    sm = summary(tr)
    sm["dets_in_tracklets"] = int(len(tdets)); sm["dropped_pct"] = round(100 * (1 - len(tr) / max(len(tdets), 1)), 2)
    unc_cam = tr.merge(dets[["det_id", "camera_id"]], on="det_id").groupby("camera_id")["uncertain"].mean()
    sm["uncertain_by_cam_pct"] = (100 * unc_cam).round(1).to_dict()
    sm["tracklets_with_jump"] = getattr(run, "split_tracklets", [])
    sm["tracklets_no_support"] = getattr(run, "no_support", [])
    sm["n_tracklets_in"] = int(tdets["tracklet_id"].nunique())
    print(json.dumps(sm, ensure_ascii=False, indent=1))
    print(f"-> {run_dir / 'trajectories.parquet'}")


if __name__ == "__main__":
    main()
