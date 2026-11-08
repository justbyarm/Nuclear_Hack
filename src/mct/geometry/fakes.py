"""Фейковые данные для склейки (У3) в формате контрактов — на нашем реальном плане.

    python -m mct.geometry.fakes --plan calib/plan.json --out outputs/fake --minutes 5 --seed 0

Синтетические люди входят через выходы плана (двери, лестницы, главный вход), идут
по опорным точкам коридоров к другому выходу, иногда стоят в холле, иногда заходят
в кабинет и выходят обратно. Пять фейковых камер с зонами обзора, похожими на наши:
тройное перекрытие в холле, слепые участки коридора 3→4 и 4→5, тупики у 339/340 и к 345.

Пишет (все пути и колонки — как в контрактах 00_ТЗ, раздел 4):
    tracklets.parquet       tracklet_id, camera_id, t_start, t_end, n_dets, local_ids (JSON-строка)
    tracklet_dets.parquet   det_id, tracklet_id
    trajectories.parquet    det_id, tracklet_id, t, x, y, sigma, zone, uncertain, vx, vy, cams_cover (JSON)
    calib/plan.json, calib/graph.json, calib/fake_fovs.json
    gt_tracklets.csv        tracklet_id, person_id, visit_id, gt_strict, gt_transit — ЭТАЛОН, в склейку не подавать
    gt_summary.json         сколько треков должно получиться в режимах strict и transit

Ошибки как в жизни: смещение каждой камеры на 10–20 см (калибровка), шум точки ног,
пропуски детекций, лишние разрывы треклетов (ID switch), сидящие с uncertain=True.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .graph import build_graph, save_graph
from .plan import Plan
from .zones import ZoneMap

# Зоны обзора фейковых камер в пикселях нашего plan_clean.png (0.0478 м/px) и их центры.
FAKE_FOVS_PX = {
    "cam1": [[222, 189], [400, 189], [400, 294], [222, 294]],
    "cam2": [[260, 189], [458, 189], [458, 294], [260, 294]],
    "cam3": [[330, 189], [458, 189], [459, 380], [415, 380], [415, 294], [330, 294]],
    "cam4": [[415, 480], [459, 480], [459, 560], [415, 560], [415, 540], [300, 540], [300, 495], [415, 495]],
    "cam5": [[415, 640], [459, 640], [459, 652], [700, 652], [700, 705], [459, 705], [459, 800], [415, 800]],
}
FAKE_CENTERS_PX = {"cam1": (360, 189), "cam2": (260, 294), "cam3": (455, 192), "cam4": (459, 520),
                   "cam5": (440, 795)}


def _nav_graph(plan: Plan, link_m=3.3):
    nodes = [np.asarray(p, float) for p in plan.nav_nodes]
    kinds = ["nav"] * len(nodes)
    for e in plan.exits:
        nodes.append(np.mean(np.asarray(e["polygon"], float), 0)); kinds.append(e["id"])
    N = np.array(nodes)
    rows, cols, w = [], [], []
    for i in range(len(N)):
        for j in range(i + 1, len(N)):
            d = np.linalg.norm(N[i] - N[j])
            if d > link_m or (kinds[i] != "nav" and kinds[j] != "nav"):
                continue
            seg = N[i] + (N[j] - N[i]) * np.linspace(0.15, 0.85, 8)[:, None]
            if plan.on_floor(seg, 0.25).all():
                rows += [i, j]; cols += [j, i]; w += [d, d]
    G = csr_matrix((w, (rows, cols)), shape=(len(N), len(N)))
    return N, kinds, G


def _path(N, G, a, b):
    dist, pred = dijkstra(G, indices=a, return_predecessors=True)
    if not np.isfinite(dist[b]):
        return None
    p, k = [b], b
    while k != a:
        k = pred[k]; p.append(k)
    return N[p[::-1]]


def _walk(points, speed, t0, dt, rng, pauses=()):
    """Положения по ломаной с постоянной скоростью; pauses: [(индекс точки, длительность)]."""
    seg = np.diff(points, axis=0)
    L = np.linalg.norm(seg, axis=1)
    pause = dict(pauses)
    out_t, out_p = [], []
    t = t0
    for i in range(len(seg)):
        if i in pause:
            n = int(pause[i] / dt)
            for _ in range(n):
                out_t.append(t); out_p.append(points[i] + rng.normal(0, 0.03, 2)); t += dt
        n = max(1, int(L[i] / speed / dt))
        for s in np.arange(n) / n:
            out_t.append(t); out_p.append(points[i] + s * seg[i]); t += dt
    out_t.append(t); out_p.append(points[-1])
    return np.array(out_t), np.array(out_p)


def simulate(plan: Plan, minutes=5.0, seed=0, fps=10.0, arrivals_per_min=6.0, t_end_s=2.0):
    rng = np.random.default_rng(seed)
    mpp = plan.m_per_px
    fovs = {c: (np.asarray(p, float) * mpp).tolist() for c, p in FAKE_FOVS_PX.items()}
    zm = ZoneMap(plan, fovs, res=0.1)
    N, kinds, G = _nav_graph(plan)
    exit_nodes = [i for i, k in enumerate(kinds) if k != "nav"]
    reach = [i for i in exit_nodes if np.isfinite(dijkstra(G, indices=i)[exit_nodes]).sum() > 1]
    hall_nodes = [i for i, k in enumerate(kinds) if k == "nav" and N[i][1] < 294 * mpp]
    dt = 1.0 / fps
    T = minutes * 60
    bias = {c: rng.normal(0, 0.12, 2) for c in fovs}
    centers = {c: np.asarray(v, float) * mpp for c, v in FAKE_CENTERS_PX.items()}

    visits = []          # (person_id, visit_id, t, xy, seated)
    pid, vid = 0, 0
    t = rng.exponential(60 / arrivals_per_min)
    while t < T - 10:
        a, b = rng.choice(reach, 2, replace=False)
        pts = _path(N, G, a, b)
        if pts is None:
            t += 1; continue
        lat = rng.normal(0, 0.25)
        d = np.diff(pts, axis=0); nrm = np.column_stack([-d[:, 1], d[:, 0]])
        nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
        pts = pts.copy(); pts[1:-1] += lat * nrm[1:]
        pauses, seated = [], False
        if rng.random() < 0.3 and hall_nodes:
            near = [k for k in range(1, len(pts) - 1) if pts[k][1] < 294 * mpp]
            if near:
                pauses.append((int(rng.choice(near)), float(rng.uniform(5, 40)))); seated = rng.random() < 0.3
        ts, ps = _walk(pts, rng.uniform(0.9, 1.5), t, dt, rng, pauses)
        visits.append((pid, vid, ts, ps, seated)); vid += 1
        # Иногда человек возвращается тем же путём (зашёл в кабинет и вышел).
        if rng.random() < 0.25:
            t_back = ts[-1] + rng.uniform(10, 90)
            if t_back < T - 10:
                ts2, ps2 = _walk(pts[::-1], rng.uniform(0.9, 1.5), t_back, dt, rng)
                visits.append((pid, vid, ts2, ps2, False)); vid += 1
        pid += 1
        t += rng.exponential(60 / arrivals_per_min)

    dets, trk_rows, gt_rows = [], [], []
    det_id, trk_id = 0, 0
    for (p_id, v_id, ts, ps, seated) in visits:
        cover = np.stack([zm.cover[k][tuple(np.clip(np.round(ps[:, ::-1] / zm.res).astype(int), 0,
                                                     np.array(zm.shape) - 1).T)] for k in range(len(zm.cam_ids))])
        # Эталон strict: новый ID, если человека не видит ни одна камера дольше T_end.
        seen = cover.any(0)
        seg_id = np.full(len(ts), -1); cur, last_seen = -1, -1e9
        for i in range(len(ts)):
            if seen[i]:
                if ts[i] - last_seen > t_end_s:
                    cur += 1
                seg_id[i] = cur; last_seen = ts[i]
        in_occ = zm.occ[tuple(np.clip(np.round(ps[:, ::-1] / zm.res).astype(int), 0, np.array(zm.shape) - 1).T)]
        for k, cam in enumerate(zm.cam_ids):
            vis = cover[k] & (rng.random(len(ts)) > 0.03) & ~(in_occ & (rng.random(len(ts)) < 0.5))
            idx = np.flatnonzero(vis)
            if len(idx) == 0:
                continue
            breaks = np.flatnonzero(np.diff(ts[idx]) > 0.5) + 1
            for chunk in np.split(idx, breaks):
                # Лишний разрыв (ID switch трекера) с вероятностью ~0.4% на кадр.
                cuts = np.flatnonzero(rng.random(len(chunk)) < 0.004)
                for part in np.split(chunk, cuts):
                    if len(part) < 5:
                        continue
                    true = ps[part]
                    dist = np.linalg.norm(true - centers[cam], axis=1)
                    sig = 0.08 + 0.025 * dist
                    unc = np.full(len(part), seated) | (rng.random(len(part)) < 0.03)
                    sig = np.where(unc, sig * 3, sig)
                    meas = true + bias[cam] + rng.normal(0, 1, true.shape) * sig[:, None]
                    k5 = np.ones(5) / 5
                    sm = np.column_stack([np.convolve(np.pad(meas[:, j], 2, mode="edge"), k5, "valid") for j in (0, 1)])
                    tt = ts[part]
                    vel = np.gradient(sm, tt, axis=0) if len(tt) > 1 else np.zeros_like(sm)
                    for r in range(len(part)):
                        dets.append((det_id, trk_id, round(float(tt[r]), 3), float(sm[r, 0]), float(sm[r, 1]),
                                     float(sig[r] / np.sqrt(3)), bool(unc[r]), float(vel[r, 0]), float(vel[r, 1])))
                        det_id += 1
                    trk_rows.append((trk_id, cam, round(float(tt[0]), 3), round(float(tt[-1]), 3), len(part),
                                     json.dumps([int(trk_id * 10 + 1)])))
                    s = seg_id[part]; s = s[s >= 0]
                    gt_rows.append((trk_id, p_id, v_id, f"{v_id}.{int(np.bincount(s).argmax()) if len(s) else 0}"))
                    trk_id += 1

    traj = pd.DataFrame(dets, columns=["det_id", "tracklet_id", "t", "x", "y", "sigma", "uncertain", "vx", "vy"])
    xy = traj[["x", "y"]].to_numpy()
    traj["zone"] = zm.zone(xy)
    traj["cams_cover"] = [json.dumps(c) for c in zm.cams_cover(xy)]
    traj = traj[["det_id", "tracklet_id", "t", "x", "y", "sigma", "zone", "uncertain", "vx", "vy", "cams_cover"]]
    trk = pd.DataFrame(trk_rows, columns=["tracklet_id", "camera_id", "t_start", "t_end", "n_dets", "local_ids"])
    gt = pd.DataFrame(gt_rows, columns=["tracklet_id", "person_id", "visit_id", "strict_key"])
    gt["gt_transit"] = gt["visit_id"]
    gt["gt_strict"] = pd.factorize(gt["strict_key"])[0]
    gt = gt[["tracklet_id", "person_id", "visit_id", "gt_strict", "gt_transit"]]
    summary = {"seed": seed, "minutes": minutes, "n_people": int(pid), "n_visits": len(visits),
               "n_tracklets": len(trk), "n_dets": len(traj),
               "expected_tracks_strict": int(gt["gt_strict"].nunique()),
               "expected_tracks_transit": int(gt["gt_transit"].nunique()),
               "t_end_s": t_end_s, "camera_bias_m": {c: np.round(b, 3).tolist() for c, b in bias.items()}}
    graph = build_graph(plan, fovs)
    return {"tracklets": trk, "tracklet_dets": traj[["det_id", "tracklet_id"]], "trajectories": traj,
            "gt": gt, "summary": summary, "graph": graph, "fovs": fovs}


def write(res: dict, out: Path, plan_path: Path, fmt="parquet"):
    out.mkdir(parents=True, exist_ok=True)
    (out / "calib").mkdir(exist_ok=True)
    for name in ("tracklets", "tracklet_dets", "trajectories"):
        df = res[name]
        if fmt == "parquet":
            df.to_parquet(out / f"{name}.parquet", index=False)
        else:
            df.to_csv(out / f"{name}.csv", index=False)
    res["gt"].to_csv(out / "gt_tracklets.csv", index=False)
    (out / "gt_summary.json").write_text(json.dumps(res["summary"], ensure_ascii=False, indent=1), encoding="utf-8")
    save_graph(res["graph"], out / "calib" / "graph.json")
    (out / "calib" / "fake_fovs.json").write_text(json.dumps(res["fovs"]), encoding="utf-8")
    shutil.copy(plan_path, out / "calib" / "plan.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="calib/plan.json")
    ap.add_argument("--out", default="outputs/fake")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    a = ap.parse_args(argv)
    plan = Plan.load(a.plan)
    res = simulate(plan, a.minutes, a.seed)
    write(res, Path(a.out), Path(a.plan), a.format)
    print(json.dumps(res["summary"], ensure_ascii=False, indent=1))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
