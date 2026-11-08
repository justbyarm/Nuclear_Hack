import json

import cv2
import numpy as np
import pandas as pd

from mct.geometry.camera import Camera, Fisheye
from mct.geometry.plan import Plan
from mct.geometry.smoothing import kalman_rts, rolling_median
from mct.geometry.trajectories import COLUMNS, run

RNG = np.random.default_rng(2)


def test_kalman_reduces_noise_and_gets_speed():
    t = np.arange(0, 8, 0.1)
    true = np.column_stack([1.0 + 1.2 * t, 2.0 + 0.5 * t])
    z = true + RNG.normal(0, 0.15, true.shape)
    xs, vs, sig, out = kalman_rts(t, rolling_median(z), np.full(len(t), 0.15))
    e_raw = np.linalg.norm(z - true, axis=1).mean()
    e_s = np.linalg.norm(xs - true, axis=1).mean()
    assert e_s < 0.5 * e_raw
    v = np.median(vs[10:-10], axis=0)
    assert np.allclose(v, [1.2, 0.5], rtol=0.1, atol=0.05)
    assert (sig < 0.15).all() and not out.any()


def test_kalman_rejects_jump_and_handles_hole():
    t = np.r_[np.arange(0, 3, 0.1), np.arange(4.0, 6, 0.1)]       # дыра 1 с
    true = np.column_stack([t, np.zeros_like(t)])
    z = true + RNG.normal(0, 0.05, true.shape)
    z[15] += [3.0, 0.0]                                            # выброс
    xs, vs, sig, out = kalman_rts(t, z, np.full(len(t), 0.05))
    assert out[15] and out.sum() <= 2
    assert np.abs(xs[:, 0] - t).max() < 0.15


def _cam():
    cam = Camera("c", (400, 300), distortion=Fisheye(300.0, 200, 150))
    cam.raw["pose"] = {"rvec": cv2.Rodrigues(np.eye(3))[0].ravel().tolist(), "center_m": [5.0, 5.0, -3.0]}
    cam.raw["fov_plan"] = [[3.5, 3.5], [6.5, 3.5], [6.5, 6.5], [3.5, 6.5]]
    cam.H = np.eye(3)
    return cam


def _head_px(cam, X, Y, h=1.7):
    R, C = cam.pose()
    P = np.column_stack([X, Y, np.full(len(X), -h)])
    c = (P - C) @ R.T
    return cam.distortion.bearings_to_pixels(c / np.linalg.norm(c, axis=1, keepdims=True))


def test_trajectories_end_to_end():
    cam = _cam()
    plan = Plan(0.01, size_m=(10.0, 10.0), floor=[[[0, 0], [10, 0], [10, 10], [0, 10]]],
                exits=[{"id": "door_a", "type": "door", "polygon": [[5.6, 4.5], [6.5, 4.5], [6.5, 5.5], [5.6, 5.5]]}])
    t = np.arange(0, 2, 0.1)
    X = 4.0 + 1.0 * t; Y = np.full_like(t, 5.0)
    uv, _ = cam.unproject(np.column_stack([X, Y]))
    uv += RNG.normal(0, 1.5, uv.shape)
    # У камеры строго вниз макушка проецируется почти в ту же точку — рамку задаём просто постоянной высоты.
    dets = pd.DataFrame({"det_id": np.arange(len(t)), "camera_id": "c", "frame_idx": np.arange(len(t)), "t": t,
                         "x1": uv[:, 0] - 8, "y1": uv[:, 1] - 60, "x2": uv[:, 0] + 8, "y2": uv[:, 1],
                         "conf": 0.9, "mask_rle": None})
    tdets = pd.DataFrame({"det_id": np.arange(len(t)), "tracklet_id": 7})
    # Камера смотрит строго вниз: высота рамки сильно меняется у надира — правило «короткой рамки» тут не к месту.
    tr = run(dets, tdets, {"c": cam}, plan, short_frac=0.0)
    assert list(tr.columns) == COLUMNS and len(tr) == len(t)
    assert np.abs(tr["x"] - X).max() < 0.1 and np.abs(tr["y"] - 5.0).max() < 0.1
    assert abs(np.median(tr["vx"]) - 1.0) < 0.15
    assert tr["uncertain"].all()                        # нет масок -> точка по рамке
    assert json.loads(tr["cams_cover"].iloc[5]) == ["c"]
    assert tr["zone"].iloc[-1].startswith("exit:door_a")


def _tilted_cam():
    """Камера на высоте 3 м, наклон 45°, смотрит вдоль +x; fisheye."""
    from mct.geometry.bundle import look_at_R
    C = np.array([0.0, 5.0, -3.0])
    R = look_at_R(C, np.array([4.0, 5.0, 0.0]))
    cam = Camera("t", (2000, 1500), distortion=Fisheye(700.0, 1000, 750, (0.02, 0, 0, 0)))
    cam.raw["pose"] = {"rvec": cv2.Rodrigues(R)[0].ravel().tolist(), "center_m": C.tolist()}
    cam.H = np.eye(3)
    return cam


def test_recover_feet_behind_table():
    """Ноги закрыты столом: низ маски — край стола (0.76 м). Без восстановления точка уезжает
    от камеры; по макушке — возвращается на место."""
    from mct.geometry.feet import recover
    cam = _tilted_cam()
    R, C = cam.pose()
    person = np.array([[4.0, 5.0], [5.0, 4.5]])
    def px(P):
        c = (P - C) @ R.T
        return cam.distortion.bearings_to_pixels(c / np.linalg.norm(c, axis=1, keepdims=True))
    edge = px(np.column_stack([person - [0.3, 0], np.full(2, -0.76)]))   # край стола перед человеком
    head = px(np.column_stack([person, np.full(2, -1.7)]))
    table = {"id": "pingpong", "center_m": [4.2, 4.8], "size_m": [1.6, 2.0], "angle_deg": 0, "height_m": 0.76}
    naive, _ = cam.project(edge)
    assert np.linalg.norm(naive - person, axis=1).min() > 0.5
    rec = recover(cam, edge, head, objects=[table])
    assert (rec["src"] == "head").all()
    assert np.linalg.norm(rec["xy"] - person, axis=1).max() < 0.05
    feet = px(np.column_stack([person, np.zeros(2)]))
    rec2 = recover(cam, feet, head, objects=[])                         # ноги видны
    assert (rec2["src"] == "foot").all() and np.allclose(rec2["vis_h"], 1.7, atol=0.05)
    seated_head = px(np.column_stack([person, np.full(2, -1.25)]))
    rec3 = recover(cam, feet, seated_head, objects=[])
    assert (rec3["src"] == "seated").all()


def test_points_never_leave_floor():
    """Часть точек ушла за горизонт/за стену — в выходе их нет за полом, они uncertain."""
    cam = _cam()
    plan = Plan(0.01, size_m=(10.0, 10.0), floor=[[[3, 3], [7, 3], [7, 7], [3, 7]]])
    t = np.arange(0, 2, 0.1)
    X = 4.0 + 1.0 * t; Y = np.full_like(t, 5.0)
    uv, _ = cam.unproject(np.column_stack([X, Y]))
    uv[5:8] = [[5, 5], [395, 5], [5, 295]]          # три точки у края кадра -> далеко за полом
    top = _head_px(cam, X, Y)
    dets = pd.DataFrame({"det_id": np.arange(len(t)), "camera_id": "c", "frame_idx": np.arange(len(t)), "t": t,
                         "x1": uv[:, 0] - 8, "y1": top[:, 1], "x2": uv[:, 0] + 8, "y2": uv[:, 1],
                         "conf": 0.9, "mask_rle": None})
    tr = run(dets, pd.DataFrame({"det_id": np.arange(len(t)), "tracklet_id": 1}), {"c": cam}, plan)
    assert tr["x"].between(3, 7).all() and tr["y"].between(3, 7).all()
    bad = tr[tr["det_id"].isin([5, 6, 7])]
    assert bad["uncertain"].all()                       # восстановлены по соседям (≤0.5 с) и помечены
    m = tr.set_index("det_id")
    assert np.abs(m["x"] - pd.Series(X)[m.index]).max() < 0.3
    d = np.hypot(np.diff(m["x"]), np.diff(m["y"])) / np.diff(m["t"])
    assert d.max() < 3.0                                 # никаких скачков


def test_jump_inside_tracklet_keeps_longest_piece():
    cam = _cam()
    plan = Plan(0.01, size_m=(10.0, 10.0), floor=[[[3, 3], [7, 3], [7, 7], [3, 7]]])
    t = np.arange(0, 2, 0.1)
    X = np.where(t < 1.2, 3.2 + 0.5 * t, 6.9); Y = np.where(t < 1.2, 6.8, 3.2)   # последние 8 точек — другой человек в 5 м
    uv, _ = cam.unproject(np.column_stack([X, Y]))
    dets = pd.DataFrame({"det_id": np.arange(len(t)), "camera_id": "c", "frame_idx": np.arange(len(t)), "t": t,
                         "x1": uv[:, 0] - 8, "y1": uv[:, 1] - 40, "x2": uv[:, 0] + 8, "y2": uv[:, 1],
                         "conf": 0.9, "mask_rle": None})
    tr = run(dets, pd.DataFrame({"det_id": np.arange(len(t)), "tracklet_id": 3}), {"c": cam}, plan, recover_feet=False)
    assert len(tr) == len(t)                         # ни одна детекция не выброшена
    d = np.hypot(np.diff(tr["x"]), np.diff(tr["y"])) / np.diff(tr["t"])
    assert d.max() <= 8.0 + 1e-6                     # в выдаче нет невозможных скачков


def test_short_box_spike_is_skipped_not_cut():
    """Человека на 2 кадра заслонили: рамка короткая, низ рамки — пояс. Треклет целый, скачка нет."""
    cam = _tilted_cam()
    R, C = cam.pose()
    def px(P):
        c = (P - C) @ R.T
        return cam.distortion.bearings_to_pixels(c / np.linalg.norm(c, axis=1, keepdims=True))
    t = np.round(np.arange(0, 3, 0.1), 2)
    X = 3.0 + 0.8 * t; Y = np.full_like(t, 5.0)
    feet = px(np.column_stack([X, Y, np.zeros_like(t)]))
    head = px(np.column_stack([X, Y, np.full_like(t, -1.7)]))
    waist = px(np.column_stack([X, Y, np.full_like(t, -0.9)]))
    bot = feet.copy(); bot[14:16] = waist[14:16]
    dets = pd.DataFrame({"det_id": np.arange(len(t)), "camera_id": "t", "frame_idx": np.arange(len(t)), "t": t,
                         "x1": bot[:, 0] - 20, "y1": head[:, 1], "x2": bot[:, 0] + 20, "y2": bot[:, 1],
                         "conf": 0.9, "mask_rle": None})
    plan = Plan(0.01, size_m=(10.0, 10.0), floor=[[[0, 0], [10, 0], [10, 10], [0, 10]]])
    tr = run(dets, pd.DataFrame({"det_id": np.arange(len(t)), "tracklet_id": 1}), {"t": cam}, plan)
    assert len(tr) == len(t)                              # треклет целый, точки восстановлены по соседям
    assert np.abs(tr["x"].to_numpy() - X).max() < 0.25
    assert tr.set_index("det_id").loc[[14, 15], "uncertain"].all()


def test_tracklet_without_support_is_kept_inside_fov():
    """Все точки треклета ненадёжны (далеко за полом): треклет всё равно в выдаче, внутри зоны камеры, uncertain."""
    cam = _cam()
    plan = Plan(0.01, size_m=(10.0, 10.0), floor=[[[4, 4], [9, 4], [9, 9], [4, 9]]])
    n = 6
    uv = np.tile([[5.0, 5.0]], (n, 1))                     # угол кадра -> проекция за горизонтом
    dets = pd.DataFrame({"det_id": np.arange(n), "camera_id": "c", "frame_idx": np.arange(n), "t": np.arange(n) * 0.1,
                         "x1": uv[:, 0], "y1": uv[:, 1] - 3, "x2": uv[:, 0] + 8, "y2": uv[:, 1],
                         "conf": 0.9, "mask_rle": None})
    tr = run(dets, pd.DataFrame({"det_id": np.arange(n), "tracklet_id": 42}), {"c": cam}, plan)
    assert len(tr) == n and tr["uncertain"].all() and (tr["sigma"] >= 1.0).all()
    assert tr["x"].between(4, 6.5).all() and tr["y"].between(4, 6.5).all()   # пол ∩ зона камеры, не угол плана


def test_clock_offsets_sign():
    from mct.geometry.overlap import clock_offsets
    off = {"cam1": 0.9, "cam3": 0.0, "cam5": 1.3}
    # пара A-B: t_B + s ≈ t_A, s = off_B − off_A
    sh = {("cam1", "cam3"): (off["cam3"] - off["cam1"], 50), ("cam3", "cam5"): (off["cam5"] - off["cam3"], 50),
          ("cam1", "cam5"): (off["cam5"] - off["cam1"], 50)}
    got = clock_offsets(sh, "cam3", ["cam1", "cam3", "cam5"])
    assert got == {"cam3": 0.0, "cam1": 0.9, "cam5": 1.3}
