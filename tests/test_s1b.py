import json

import numpy as np
import pandas as pd
import pytest

from mct.geometry import rle as R
from mct.geometry.camera import Camera, Fisheye
from mct.geometry.floor_check import run
from mct.geometry.foot import foot_point
from mct.geometry.plan import Plan

RNG = np.random.default_rng(1)

def test_rle_roundtrip_random_masks():
    for _ in range(30):
        H, W = RNG.integers(5, 80, 2)
        m = RNG.random((H, W)) > RNG.uniform(0.2, 0.9)
        r = R.encode(m)
        assert isinstance(r["counts"], str)
        assert (R.decode(r) == m).all()
        assert (R.decode(json.dumps(r)) == m).all()


def test_rle_column_major_runs():
    # Маска 3x4, развёрнутая по столбцам: 0000 11 0 1 0000 -> runs 4,2,1,1,4 (как в COCO).
    m = np.zeros((3, 4), bool); m[1, 1] = m[2, 1] = m[1, 2] = True
    H, W, counts = R.parse(R.encode(m))
    assert (H, W) == (3, 4) and counts.tolist() == [4, 2, 1, 1, 4]
    assert (R.decode({"size": [3, 4], "counts": [4, 2, 1, 1, 4]}) == m).all()
    # Отрицательная разность в строковом кодировании (x - cnts[i-2] < 0) тоже обратима.
    big = np.zeros((50, 50), bool); big[10:40, 5] = True; big[12, 6] = True
    assert (R.decode(R.encode(big)) == big).all()


def test_bottom_info_matches_full_decode():
    for _ in range(40):
        H, W = 120, 90
        m = np.zeros((H, W), bool)
        cx, cy = RNG.integers(20, 70), RNG.integers(20, 90)
        yy, xx = np.ogrid[:H, :W]
        m |= ((xx - cx) / RNG.uniform(5, 15)) ** 2 + ((yy - cy) / RNG.uniform(10, 25)) ** 2 < 1
        m[RNG.integers(0, H, 5), RNG.integers(0, W, 5)] = True       # шум
        yb, xm, yt, area = R.bottom_info(R.encode(m), rows=4)
        ys, xs = np.nonzero(m)
        assert yb == ys.max() and yt == ys.min() and area == m.sum()
        sel = ys >= yb - 3
        ref = np.sort(xs[sel])
        assert ref[0] <= xm <= ref[-1]
        assert xm in (ref[(len(ref) - 1) // 2], ref[len(ref) // 2])


def test_foot_point_fallbacks():
    m = np.zeros((100, 60), bool); m[20:90, 25:35] = True
    u, v, src, cut = foot_point(json.dumps(R.encode(m)), (20, 18, 40, 92), (60, 100))
    assert src == "mask" and v == 89 and 25 <= u <= 34 and not cut
    u, v, src, cut = foot_point(None, (20, 18, 40, 92), (60, 100))
    assert src == "bbox" and (u, v) == (30.0, 92.0)
    m[20:100, 25:35] = True
    *_, cut = foot_point(R.encode(m), (20, 18, 40, 99), (60, 100))
    assert cut


def _overhead_cam(W=400, H=300):
    """Пинхол-камера ровно вниз над точкой (5, 5) на высоте 3 м; 1 px = 1 см у центра."""
    cam = Camera("c", (W, H), distortion=Fisheye(300.0, W / 2, H / 2))
    import cv2
    R_ = np.diag([1.0, 1.0, 1.0])            # камера z = мировое z (вниз, в пол)
    cam.raw["pose"] = {"rvec": cv2.Rodrigues(R_)[0].ravel().tolist(), "center_m": [5.0, 5.0, -3.0]}
    cam.H = np.eye(3)                         # не используется: есть поза
    return cam


def test_floor_check_rejects_points_outside_floor():
    cam = _overhead_cam()
    plan = Plan(0.01, floor=[[[4.6, 4.6], [5.4, 4.6], [5.4, 5.4], [4.6, 5.4]]])
    rows = []
    for i, (u, v) in enumerate([(200, 150), (390, 150), (200, 290)]):
        rows.append({"det_id": i, "camera_id": "c", "x1": u - 5, "y1": v - 40, "x2": u + 5, "y2": v,
                     "mask_rle": None})
    fc = run(pd.DataFrame(rows), {"c": cam}, plan, tol_m=0.1)
    x = fc["plan_x_raw"].to_numpy()
    assert abs(x[0] - 5.0) < 1e-6
    assert fc["on_floor"].tolist() == [True, False, False]
    assert list(fc.columns)[:7] == ["det_id", "foot_u", "foot_v", "plan_x_raw", "plan_y_raw", "on_floor", "in_fov"]


def test_floor_check_uncalibrated_camera_keeps_everything():
    plan = Plan(0.01, floor=[[[0, 0], [1, 0], [1, 1], [0, 1]]])
    d = pd.DataFrame([{"det_id": 7, "camera_id": "x", "x1": 0, "y1": 0, "x2": 10, "y2": 10, "mask_rle": None}])
    fc = run(d, {}, plan)
    assert fc["on_floor"].all() and fc["plan_x_raw"].isna().all()
