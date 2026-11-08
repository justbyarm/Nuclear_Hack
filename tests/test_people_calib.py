"""Связка камер по людям: пары «один человек на двух камерах» -> D_max и уточнение калибровки."""
import numpy as np
import pandas as pd

from mct.geometry.bundle import Bundle, CamObs, cam_params_from_camera, refine_with_people, to_camera
from mct.geometry.overlap import find_pairs, stats, time_shift
from mct.geometry.plan import Plan
from mct.geometry.zones import ZoneMap

from test_bundle import CAM1, CAM2, MPP, floor_err, make_cam, observe

PLAN = Plan(MPP, size_m=(11.0, 5.0), floor=[[[0, 0], [11, 0], [11, 5], [0, 5]]])


def _cams_true():
    out = {}
    for name, spec in (("cam1", CAM1), ("cam2", CAM2)):
        m, R, C, size = make_cam(**spec)
        b = Bundle([CamObs.from_json({"camera_id": name, "image_size": list(size)}, MPP)], MPP)
        cam = to_camera(b, b.pack_cam(m, R, C), 0)
        cam.raw["fov_plan"] = [[0, 0], [11, 0], [11, 5], [0, 5]]
        out[name] = (cam, spec)
    return out


def _people(cams, rng, n_people=6, T=40.0, dt_a=0.04, dt_b=0.05, shift_b=0.0):
    """Люди ходят по холлу; каждая камера видит их со своим fps. Шум точки ног 2 px."""
    rows, det = [], 0
    for p in range(n_people):
        w = rng.uniform([1, 0.8], [10, 4.2], (6, 2))
        t0 = p * T / n_people
        for cam_name, dt, sh in (("cam1", dt_a, 0.0), ("cam2", dt_b, shift_b)):
            cam = cams[cam_name][0]
            t = np.arange(t0, t0 + 6.0, dt)
            s = (t - t0) / 6.0 * 5
            xy = np.column_stack([np.interp(s, np.arange(6), w[:, 0]), np.interp(s, np.arange(6), w[:, 1])])
            uv, ok = cam.unproject(xy)
            for k in np.flatnonzero(ok):
                u, v = uv[k] + rng.normal(0, 2, 2)
                rows.append((det, cam_name, round(float(t[k] - sh), 3), u, v)); det += 1
    d = pd.DataFrame(rows, columns=["det_id", "camera_id", "t", "foot_u", "foot_v"])
    return d


def _fc(d, cams):
    xs, ys = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for c, (cam, _) in cams.items():
        m = (d["camera_id"] == c).to_numpy()
        xy, _ = cam.project(d.loc[m, ["foot_u", "foot_v"]].to_numpy())
        xs[m], ys[m] = xy[:, 0], xy[:, 1]
    return d.assign(x=xs, y=ys)


def test_pairs_dmax_and_time_shift():
    rng = np.random.default_rng(0)
    cams = _cams_true()
    pts = _fc(_people(cams, rng, shift_b=0.3), cams)
    zm = ZoneMap(PLAN, {c: v[0].raw["fov_plan"] for c, v in cams.items()})
    s = time_shift(pts, zm, "cam1", "cam2", step=0.05)
    assert s is not None and abs(s[0] - 0.3) <= 0.1, s
    p = find_pairs(pts, zm, "cam1", "cam2", shift=s[0])
    st = stats(p)
    assert st["n"] > 100 and st["median_m"] < 0.15, st
    p0 = find_pairs(pts, zm, "cam1", "cam2", shift=0.0)
    assert stats(p0)["median_m"] > st["median_m"]          # без поправки времени хуже


def test_refine_fixes_camera_without_plan_anchors():
    """cam2 откалибрована с ошибкой ~0.7 м и 3°; плана у неё нет (как у камеры 5).
    Пары с людьми, видимыми cam1, должны вернуть её на место."""
    rng = np.random.default_rng(1)
    cams = _cams_true()
    pts = _fc(_people(cams, rng, n_people=10), cams)
    zm = ZoneMap(PLAN, {c: v[0].raw["fov_plan"] for c, v in cams.items()})
    pairs = find_pairs(pts, zm, "cam1", "cam2")
    # Наблюдения калибровки: cam1 — стены, плитка, стол; cam2 — только прямые и плитка.
    m1, R1, C1, size = make_cam(**CAM1)
    m2, R2, C2, _ = make_cam(**CAM2)
    o1 = observe("cam1", m1, R1, C1, size, rng); o1["hint"] = {}
    o2 = observe("cam2", m2, R2, C2, size, rng); o2.update(walls=[], objects=[], landmarks=[], pairs=[], hint={})
    init = {"cam1": cam_params_from_camera(cams["cam1"][0]), "cam2": cam_params_from_camera(cams["cam2"][0])}
    bad = init["cam2"].copy()
    bad[-3:] += [0.6, -0.4, 0.0]              # центр сдвинут
    bad[5:8] += np.deg2rad([2.0, -2.0, 3.0])  # поворот
    init["cam2"] = bad
    b0 = Bundle([CamObs.from_json(o2, MPP)], MPP)
    err_before = np.percentile(floor_err(to_camera(b0, bad, 0), m2, R2, C2, size, rng), 95)
    b, p = refine_with_people({"cam1": o1, "cam2": o2}, init, pairs, MPP, n_max=150)
    i2 = [c.camera_id for c in b.cams].index("cam2")
    err_after = np.percentile(floor_err(to_camera(b, p, i2), m2, R2, C2, size, rng), 95)
    assert err_before > 0.5 and err_after < 0.12, (err_before, err_after)
