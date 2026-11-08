"""Один человек на двух камерах одновременно: пары наблюдений в зонах перекрытия.

    python -m mct.geometry.overlap --run outputs/run --calib calib

Из пар считается:
  - D_max для склейки (У3): 95-й процентиль расстояния между точками одного человека;
  - нормированное расстояние d / sqrt(σa² + σb²) — один порог на все пары камер;
  - систематическое смещение пары камер (средний вектор разности) — признак плохой калибровки;
  - сдвиг времени между камерами (перебор Δt, минимум медианного расстояния);
  - пары для уточнения калибровки по людям (refine_with_people в bundle).

Однозначность: берём моменты, когда в зоне перекрытия ровно ОДНА точка с каждой камеры.
Разметка не нужна.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .floor_check import load_cameras
from .plan import Plan
from .zones import ZoneMap


def _points(dets: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame:
    d = dets[["det_id", "camera_id", "t"]].merge(
        fc[["det_id", "foot_u", "foot_v", "plan_x_raw", "plan_y_raw", "on_floor"]
           + [c for c in ("cut_bottom", "foot_src") if c in fc.columns]], on="det_id")
    d = d[d["on_floor"] & np.isfinite(d["plan_x_raw"])]
    if "cut_bottom" in d:
        d = d[~d["cut_bottom"].astype(bool)]
    if "foot_src" in d:
        d = d[d["foot_src"] == "mask"]
    return d.rename(columns={"plan_x_raw": "x", "plan_y_raw": "y"})


def find_pairs(pts: pd.DataFrame, zm: ZoneMap, cam_a: str, cam_b: str, dt_tol: float = 0.06,
               shift: float = 0.0, max_dist_m: float = 3.0) -> pd.DataFrame:
    """Пары (A, B): в момент t в перекрытии A∩B ровно одна точка A и ровно одна точка B."""
    ia, ib = zm.cam_ids.index(cam_a), zm.cam_ids.index(cam_b)
    both = zm.cover[ia] & zm.cover[ib]
    out = []
    sides = {}
    for cam in (cam_a, cam_b):
        p = pts[pts["camera_id"] == cam]
        ij, ok = zm._ij(p[["x", "y"]].to_numpy())
        # Небольшой допуск: точка у края перекрытия (ошибка калибровки) всё ещё считается.
        inside = ok & both[ij[:, 0], ij[:, 1]]
        p = p[inside]
        # Ровно одна точка в перекрытии в этот кадр этой камеры.
        cnt = p.groupby("t")["det_id"].transform("size")
        sides[cam] = p[cnt == 1].sort_values("t")
    A, B = sides[cam_a], sides[cam_b]
    if A.empty or B.empty:
        return pd.DataFrame()
    tb = B["t"].to_numpy() + shift
    k = np.clip(np.searchsorted(tb, A["t"].to_numpy()), 1, len(tb) - 1)
    kk = np.where(np.abs(tb[k] - A["t"].to_numpy()) < np.abs(tb[k - 1] - A["t"].to_numpy()), k, k - 1)
    good = np.abs(tb[kk] - A["t"].to_numpy()) <= dt_tol
    Aa, Bb = A[good].reset_index(drop=True), B.iloc[kk[good]].reset_index(drop=True)
    d = np.hypot(Aa["x"] - Bb["x"], Aa["y"] - Bb["y"])
    keep = d <= max_dist_m
    res = pd.DataFrame({"t": Aa["t"], "cam_a": cam_a, "cam_b": cam_b,
                        "det_a": Aa["det_id"], "det_b": Bb["det_id"],
                        "u_a": Aa["foot_u"], "v_a": Aa["foot_v"], "u_b": Bb["foot_u"], "v_b": Bb["foot_v"],
                        "xa": Aa["x"], "ya": Aa["y"], "xb": Bb["x"], "yb": Bb["y"], "dist": d})
    return res[keep.to_numpy()].reset_index(drop=True)


def time_shift(pts, zm, cam_a, cam_b, rng=(-1.0, 1.0), step=0.1, min_pairs=30, **kw):
    """Сдвиг B относительно A: t_B + shift ≈ t_A. -> (shift, медиана расстояния, число пар)."""
    best = None
    for s in np.round(np.arange(rng[0], rng[1] + 1e-9, step), 3):
        p = find_pairs(pts, zm, cam_a, cam_b, shift=s, **kw)
        if len(p) < min_pairs:
            continue
        m = float(p["dist"].median())
        if best is None or m < best[1]:
            best = (float(s), m, len(p))
    return best


def stats(pairs: pd.DataFrame, sig_a=None, sig_b=None) -> dict:
    if pairs.empty:
        return {"n": 0}
    d = pairs["dist"].to_numpy()
    dx, dy = (pairs["xb"] - pairs["xa"]).to_numpy(), (pairs["yb"] - pairs["ya"]).to_numpy()
    out = {"n": int(len(d)), "median_m": round(float(np.median(d)), 3),
           "p95_m": round(float(np.percentile(d, 95)), 3),
           "bias_m": [round(float(np.median(dx)), 3), round(float(np.median(dy)), 3)]}
    if sig_a is not None and sig_b is not None:
        z = d / np.sqrt(np.asarray(sig_a) ** 2 + np.asarray(sig_b) ** 2)
        out["within_2sigma_pct"] = round(100 * float((z < 2).mean()), 1)
        out["z_p95"] = round(float(np.percentile(z, 95)), 2)
    return out


def analyze(dets, fc, cams, plan, shift_search=True) -> tuple[dict, pd.DataFrame]:
    pts = _points(dets, fc)
    fovs = {c: cams[c].raw.get("fov_plan") for c in cams if cams[c].raw.get("fov_plan")}
    zm = ZoneMap(plan, fovs)
    report, allp = {"pairs": {}}, []
    ids = zm.cam_ids
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if not (zm.cover[i] & zm.cover[j]).any():
                continue
            rec = {}
            shift = 0.0
            if shift_search:
                ts = time_shift(pts, zm, a, b)
                if ts:
                    shift = ts[0]
                    rec["time_shift_s"] = shift
            p = find_pairs(pts, zm, a, b, shift=shift)
            if p.empty:
                continue
            mpp = {c: cams[c].meters_per_pixel(p[[f"u_{s}", f"v_{s}"]].to_numpy()) for c, s in ((a, "a"), (b, "b"))}
            sa = 0.15 + 3.0 * np.nan_to_num(mpp[a], nan=0.05, posinf=0.5)
            sb = 0.15 + 3.0 * np.nan_to_num(mpp[b], nan=0.05, posinf=0.5)
            rec.update(stats(p, sa, sb))
            report["pairs"][f"{a}-{b}"] = rec
            allp.append(p)
    allp = pd.concat(allp, ignore_index=True) if allp else pd.DataFrame()
    if len(allp):
        report["D_max_m"] = round(float(np.percentile(allp["dist"], 95)), 3)
        report["n_pairs"] = int(len(allp))
    return report, allp


def _traj_points(tr: pd.DataFrame, dets: pd.DataFrame, fc: pd.DataFrame | None, only_certain=True,
                 max_sigma=1.0) -> pd.DataFrame:
    """Сглаженные точки траекторий (S5) — именно их сравнивает склейка У3.
    only_certain: только надёжные; иначе все, кроме «удержанных» (σ ≥ max_sigma)."""
    m = ~tr["uncertain"].astype(bool) if only_certain else tr["sigma"] < max_sigma
    d = tr[m][["det_id", "t", "x", "y", "sigma"]]
    d = d.merge(dets[["det_id", "camera_id"]], on="det_id")
    if fc is not None and "foot_u" in fc.columns:
        d = d.merge(fc[["det_id", "foot_u", "foot_v"]], on="det_id", how="left")
    else:
        d["foot_u"] = np.nan; d["foot_v"] = np.nan
    return d


def clock_offsets(shifts: dict, ref: str, cams: list) -> dict:
    """Сдвиги часов по камерам из сдвигов пар (взвешенный МНК, ref = 0).

    Пара «A-B» со сдвигом s: t_B + s ≈ t_A. Если off_c — поправка камеры c (t_c + off_c = время ref),
    то s = off_B − off_A.
    """
    others = [c for c in cams if c != ref]
    rows, rhs, w = [], [], []
    for (a, b), (s, n) in shifts.items():
        r = np.zeros(len(others))
        if b != ref: r[others.index(b)] += 1
        if a != ref: r[others.index(a)] -= 1
        rows.append(r); rhs.append(s); w.append(np.sqrt(n))
    out = {ref: 0.0}
    if not rows:
        return out
    A = np.array(rows) * np.array(w)[:, None]; y = np.array(rhs) * np.array(w)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    seen = np.abs(np.array(rows)).sum(0) > 0
    for c, v, ok in zip(others, sol, seen):
        out[c] = round(float(v), 2) if ok else None
    return out


MIN_PAIRS = 30


def analyze_traj(tr, dets, fc, cams, plan, ref="cam3", rng=(-2.0, 2.0), step=0.05) -> tuple[dict, pd.DataFrame]:
    """Как analyze, но по траекториям S5 (сглаженные надёжные точки) + сдвиги часов по камерам."""
    pts_c = _traj_points(tr, dets, fc)
    pts_a = _traj_points(tr, dets, fc, only_certain=False)
    fovs = {c: cams[c].raw.get("fov_plan") for c in cams if cams[c].raw.get("fov_plan")}
    zm = ZoneMap(plan, fovs)
    sig = pts_a.set_index("det_id")["sigma"]
    report, allp, shifts = {"source": "trajectories", "pairs": {}}, [], {}
    ids = zm.cam_ids
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if not (zm.cover[i] & zm.cover[j]).any():
                continue
            # Сначала только надёжные точки; если пар мало — все, кроме удержанных (σ ≥ 1 м).
            used = "certain"
            ts = time_shift(pts_c, zm, a, b, rng=rng, step=step, min_pairs=MIN_PAIRS)
            pts = pts_c
            if not ts:
                ts = time_shift(pts_a, zm, a, b, rng=rng, step=step, min_pairs=MIN_PAIRS)
                pts, used = pts_a, "all_sigma_lt_1m"
            shift = ts[0] if ts else 0.0
            p = find_pairs(pts, zm, a, b, shift=shift)
            if p.empty:
                report["pairs"][f"{a}-{b}"] = {"n": 0, "reliable": False}
                continue
            rec = {"time_shift_s": round(shift, 2)} if ts else {}
            rec["points"] = used
            rec.update(stats(p, sig.loc[p["det_a"]].to_numpy(), sig.loc[p["det_b"]].to_numpy()))
            rec["D_max_m"] = rec["p95_m"]
            rec["reliable"] = bool(ts) and len(p) >= MIN_PAIRS
            report["pairs"][f"{a}-{b}"] = rec
            if ts:
                shifts[(a, b)] = (shift, len(p))
            allp.append(p)
    allp = pd.concat(allp, ignore_index=True) if allp else pd.DataFrame()
    if len(allp):
        report["D_max_m"] = round(float(np.percentile(allp["dist"], 95)), 3)
        report["n_pairs"] = int(len(allp))
    off = clock_offsets(shifts, ref, ids)
    report["clock_offset_s"] = off
    rel = [k for k, v in report["pairs"].items() if v.get("reliable")]
    report["note"] = ("D_max_m по паре надёжен, если reliable=true (≥ 30 пар). Для остальных пар брать "
                      "общий D_max_m. clock_offset_s = null — сдвиг этой камеры по данным не определяется.")
    report["reliable_pairs"] = rel
    report["clock_note"] = (f"t_cam + clock_offset_s[cam] = время {ref}. "
                            "В паре A-B: t_B + time_shift_s ≈ t_A (time_shift_s ≈ off_B − off_A).")
    # Насколько согласованы сдвиги пар с поправками камер (невязка, с).
    report["clock_residual_s"] = {f"{a}-{b}": round(s - ((off.get(b) or 0) - (off.get(a) or 0)), 2)
                                  for (a, b), (s, n) in shifts.items()}
    return report, allp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--calib", default="calib")
    ap.add_argument("--raw", action="store_true", help="по сырым точкам ног (как раньше), а не по траекториям")
    ap.add_argument("--ref", default="cam3", help="камера, относительно которой считаются сдвиги часов")
    a = ap.parse_args(argv)
    run = Path(a.run)
    dets = pd.read_parquet(run / "detections.parquet")
    fcp = run / "floor_check.parquet"
    fc = pd.read_parquet(fcp) if fcp.exists() else None
    plan = Plan.load(Path(a.calib) / "plan.json")
    cams = load_cameras(a.calib)
    if a.raw:
        rep, pairs = analyze(dets, fc, cams, plan)
    else:
        tr = pd.read_parquet(run / "trajectories.parquet")
        rep, pairs = analyze_traj(tr, dets, fc, cams, plan, ref=a.ref)
    pairs.to_parquet(run / "overlap_pairs.parquet", index=False)
    (run / "overlap_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
