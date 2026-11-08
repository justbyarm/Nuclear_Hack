"""Bundle-калибровка на синтетике: холл с двумя камерами, стенами и теннисным столом."""
import numpy as np
import pytest

from mct.geometry.bundle import (OBJECTS, Bundle, CamObs, backproject_floor, calibrate_single,
                                 look_at_R, object_corners, project, report, to_camera)
from mct.geometry.camera import Fisheye

MPP = 0.05                       # план: 5 см на пиксель
HALL = (0.0, 0.0, 11.0, 5.0)     # x0 y0 x1 y1, метры
TABLE = np.array([5.0, 2.6, np.deg2rad(8)])


def make_cam(C, target, model, size=(2592, 1944)):
    C = np.asarray(C, float)
    return model, look_at_R(C, target), C, size


def visible(model, R, C, P, size):
    uv, fr = project(model, R, C, P)
    W, H = size
    ok = fr & (uv[:, 0] > 5) & (uv[:, 0] < W - 5) & (uv[:, 1] > 5) & (uv[:, 1] < H - 5)
    b = (np.asarray(P, float).reshape(-1, 3) - C) @ R.T
    ang = np.degrees(np.arccos(b[:, 2] / np.linalg.norm(b, axis=1)))
    return uv, ok & (ang < 82)


def observe(name, model, R, C, size, rng, noise=1.0, n_pairs=0, landmarks=()):
    x0, y0, x1, y1 = HALL
    lines, walls = [], []
    # Стены: юг (y=y1) и запад (x=x0) и восток (x=x1) — точки низа стены.
    for a, b in [((x0, y1), (x1, y1)), ((x0, y0), (x0, y1)), ((x1, y0), (x1, y1)), ((x0, y0), (x1, y0))]:
        t = np.linspace(0, 1, 40)
        P = np.column_stack([np.outer(1 - t, a) + np.outer(t, b), np.zeros(len(t))])
        uv, ok = visible(model, R, C, P, size)
        if ok.sum() >= 6:
            pts = uv[ok] + rng.normal(0, noise, (ok.sum(), 2))
            walls.append({"img": pts.tolist(), "plan_px": (np.array([a, b]) / MPP).tolist()})
            lines.append(pts.tolist())
    # Швы плитки 0.6 м.
    for x in np.arange(0.6, x1, 1.2):
        P = np.column_stack([np.full(25, x), np.linspace(0.2, y1 - 0.2, 25), np.zeros(25)])
        uv, ok = visible(model, R, C, P, size)
        if ok.sum() >= 6:
            lines.append((uv[ok] + rng.normal(0, noise, (ok.sum(), 2))).tolist())
    for y in np.arange(0.6, y1, 1.2):
        P = np.column_stack([np.linspace(0.2, x1 - 0.2, 40), np.full(40, y), np.zeros(40)])
        uv, ok = visible(model, R, C, P, size)
        if ok.sum() >= 6:
            lines.append((uv[ok] + rng.normal(0, noise, (ok.sum(), 2))).tolist())
    objs = []
    uv, ok = visible(model, R, C, object_corners(OBJECTS["pingpong"], TABLE), size)
    if ok.all():
        objs.append({"id": "pingpong", "img": (uv + rng.normal(0, noise, uv.shape)).tolist()})
    lms = []
    for lid, xy in landmarks:
        uv, ok = visible(model, R, C, [[*xy, 0.0]], size)
        if ok[0]:
            lms.append({"id": lid, "img": (uv[0] + rng.normal(0, noise, 2)).tolist()})
    pairs = []
    for _ in range(n_pairs):
        xy = rng.uniform([x0 + 0.5, y0 + 0.5], [x1 - 0.5, y1 - 0.5])
        uv, ok = visible(model, R, C, [[*xy, 0.0]], size)
        if ok[0]:
            pairs.append({"img": (uv[0] + rng.normal(0, noise, 2)).tolist(), "plan_px": (xy / MPP).tolist()})
    return {"camera_id": name, "image_size": list(size), "lines": lines, "walls": walls,
            "objects": objs, "landmarks": lms, "pairs": pairs}


def floor_err(cam, model, R, C, size, rng, n=3000):
    x0, y0, x1, y1 = HALL
    XY = np.column_stack([rng.uniform(x0 + 0.2, x1 - 0.2, n), rng.uniform(y0 + 0.2, y1 - 0.2, n)])
    uv, ok = visible(model, R, C, np.column_stack([XY, np.zeros(n)]), size)
    pred, good = cam.project(uv[ok])
    assert good.all()
    return np.linalg.norm(pred - XY[ok], axis=1)


CAM1 = dict(C=(5.0, 0.0, -2.9), target=(4.0, 4.0, 0.0),
            model=Fisheye(900.0, 1300.0, 960.0, (0.03, -0.004, 0, 0)))
CAM2 = dict(C=(0.2, 4.9, -2.7), target=(5.0, 1.5, 0.0),
            model=Fisheye(950.0, 1290.0, 980.0, (-0.02, 0.003, 0, 0)))


def test_to_camera_matches_backprojection():
    m, R, C, size = make_cam(**CAM1)
    obs = CamObs.from_json({"camera_id": "c", "image_size": list(size), "hint": {}}, MPP)
    b = Bundle([obs], MPP)
    p = b.pack_cam(m, R, C)
    cam = to_camera(b, p, 0)
    uv = np.array([[1300, 1500], [600, 1200], [2000, 1700]], float)
    X, ok = backproject_floor(m, R, C, uv)
    Y, ok2 = cam.project(uv)
    assert ok.all() and ok2.all() and np.abs(X - Y).max() < 1e-6
    assert abs(cam.raw["pose"]["height_m"] - 2.9) < 1e-9


@pytest.mark.parametrize("seed", range(3))
def test_single_camera_walls_and_table_no_pairs(seed):
    """Ни одной пары кадр↔план: только стены, плитка и стол известного размера."""
    rng = np.random.default_rng(seed)
    m, R, C, size = make_cam(**CAM1)
    d = observe("cam1", m, R, C, size, rng)
    assert d["objects"] and not d["pairs"]
    d["hint"] = {"center_px": [5.4 / MPP, 0.3 / MPP], "look_at_px": [3.5 / MPP, 3.0 / MPP], "height_m": 2.8}
    b, p = calibrate_single(CamObs.from_json(d, MPP), MPP)
    rep = report(b, p)
    cam = to_camera(b, p, 0)
    err = floor_err(cam, m, R, C, size, rng)
    assert np.percentile(err, 95) < 0.12, rep
    assert abs(rep["cam1"]["height_m"] - 2.9) < 0.1, rep


def test_two_cameras_share_table_and_landmarks():
    rng = np.random.default_rng(5)
    lms = [("stand", (1.2, 4.2)), ("foos", (9.5, 4.0)), ("leg", (6.0, 3.4))]
    obs = []
    for name, spec, hint in [("cam1", CAM1, ((5.4, 0.3), (3.5, 3.0))), ("cam2", CAM2, ((0.6, 4.5), (5.0, 2.0)))]:
        m, R, C, size = make_cam(**spec)
        d = observe(name, m, R, C, size, rng, landmarks=lms)
        d["hint"] = {"center_px": (np.array(hint[0]) / MPP).tolist(),
                     "look_at_px": (np.array(hint[1]) / MPP).tolist(), "height_m": 2.8}
        obs.append(CamObs.from_json(d, MPP))
    # Каждая камера отдельно -> старт для совместного решения.
    starts = [calibrate_single(o, MPP) for o in obs]
    b = Bundle(obs, MPP)
    p = b.init_params()
    for i, (bi, pi) in enumerate(starts):
        p[b.cam_slice(i)] = pi[bi.cam_slice(0)]
    p = b.init_params_objects_from(p) if hasattr(b, "init_params_objects_from") else p
    r = b.solve(p)
    rep = report(b, r.x)
    assert len(b.lm_ids) >= 2, b.lm_ids
    for i, spec in enumerate([CAM1, CAM2]):
        m, R, C, size = make_cam(**spec)
        err = floor_err(to_camera(b, r.x, i), m, R, C, size, rng)
        assert np.percentile(err, 95) < 0.12, rep
    t = r.x[b.obj_slice(0)]
    assert np.linalg.norm(t[:2] - TABLE[:2]) < 0.1, rep
