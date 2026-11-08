import cv2
import numpy as np
import pytest

from mct.geometry import polygons as P
from mct.geometry.calibrate import calibrate_camera
from mct.geometry.camera import Camera, Fisheye
from mct.geometry.distortion import fit_fisheye_from_lines
from mct.geometry.fit import fit_homography
from mct.geometry.plan import scale_from_segments

from synthetic import SynthCam

RNG = np.random.default_rng(0)


def test_fisheye_matches_opencv():
    m = Fisheye(650.0, 1280.0, 960.0, (-0.03, 0.004, 0.001, 0.0))
    K = np.array([[m.f, 0, m.cx], [0, m.f, m.cy], [0, 0, 1]])
    D = np.array(m.k)
    xy = RNG.uniform(-2, 2, (200, 2))
    ours = m.distort(xy)
    ref = cv2.fisheye.distortPoints(xy.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    assert np.abs(ours - ref).max() < 1e-6


def test_fisheye_roundtrip_beyond_90deg():
    m = Fisheye(600.0, 1000.0, 800.0, (-0.02, 0.001, 0, 0))
    th = np.deg2rad(np.array([0, 30, 60, 89, 95, 100]))
    b = np.stack([np.sin(th), np.zeros_like(th), np.cos(th)], 1)
    back = m.pixels_to_bearings(m.bearings_to_pixels(b))
    assert np.abs(back - b).max() < 1e-9


def _true_floor_pairs(cam, n=14):
    XY = np.column_stack([RNG.uniform(1, 9, 200), RNG.uniform(0.5, 8, 200)])
    uv, vis = cam.floor_to_pixels(XY)
    return uv[vis][:n], XY[vis][:n]


def test_lines_get_straight():
    sc = SynthCam()
    lines = [l + RNG.normal(0, 0.7, l.shape) for l in sc.floor_lines()]   # дрожание кликов
    fit = fit_fisheye_from_lines(lines, sc.size)
    assert fit.rms_before_px > 10 * fit.rms_after_px                      # линии реально изогнуты
    assert fit.rms_after_px < 1.5


def _floor_error(sc, cam, rng, n=3000):
    T = np.column_stack([rng.uniform(0, 10, n), rng.uniform(0.3, 9, n)])
    tuv, vis = sc.floor_to_pixels(T)
    pred, ok = cam.project(tuv[vis])
    assert ok.all()
    return np.linalg.norm(pred - T[vis], axis=1)


@pytest.mark.parametrize("seed", range(4))
def test_joint_calibration_extrapolates_to_frame_edges(seed):
    """Прямые кликнуты только в центре, пары — средней кучностью, шум кликов 1 px.
    Проверяем ошибку по ВСЕМУ видимому полу, включая края кадра."""
    rng = np.random.default_rng(seed)
    sc = SynthCam()
    lines = [l + rng.normal(0, 0.7, l.shape) for l in sc.floor_lines(xs=(4, 5, 6), ys=(2, 3, 4))]
    XY = np.column_stack([rng.uniform(1.5, 8.5, 200), rng.uniform(1, 7, 200)])
    uv, vis = sc.floor_to_pixels(XY)
    uv, XY = uv[vis][:12], XY[vis][:12]
    res = calibrate_camera("synth", sc.size, uv + rng.normal(0, 1.0, uv.shape), XY, lines)
    err = _floor_error(sc, res.camera, rng)
    assert np.percentile(err, 95) < 0.12, res.summary()
    assert abs(res.height_m - 2.8) < 0.1
    assert abs(res.tilt_deg - 45) < 2


@pytest.mark.parametrize("use_lines", [True, False])
def test_joint_calibration_strong_fisheye(use_lines):
    """Сильнее «рыбий глаз» (≈ как на офисных кадрах), камера ниже и под углом."""
    rng = np.random.default_rng(11)
    sc = SynthCam(pos=(1.0, 0.5, 2.6), pitch_deg=38, yaw_deg=25,
                  model=Fisheye(520.0, 1270.0, 975.0, (0.02, -0.004, 0, 0)))
    XY = np.column_stack([rng.uniform(0, 9, 400), rng.uniform(0.5, 9, 400)])
    uv, vis = sc.floor_to_pixels(XY)
    uv, XY = uv[vis][:14], XY[vis][:14]
    lines = [l + rng.normal(0, 0.7, l.shape) for l in sc.floor_lines()] if use_lines else ()
    res = calibrate_camera("synth", sc.size, uv + rng.normal(0, 1.0, uv.shape), XY, lines)
    err = _floor_error(sc, res.camera, rng)
    # Без прямых дальний пол хуже: прямые кликать обязательно.
    assert np.percentile(err, 95) < (0.12 if use_lines else 0.35), res.summary()
    assert abs(res.height_m - 2.6) < 0.15


def test_joint_calibration_survives_misclick():
    rng = np.random.default_rng(7)
    sc = SynthCam()
    XY = np.column_stack([rng.uniform(1.5, 8.5, 200), rng.uniform(1, 7, 200)])
    uv, vis = sc.floor_to_pixels(XY)
    uv, XY = uv[vis][:12], XY[vis][:12].copy()
    XY[5] += [1.2, 0.8]
    res = calibrate_camera("synth", sc.size, uv, XY, sc.floor_lines())
    assert not res.inliers[5] and res.inliers.sum() == 11
    assert np.percentile(_floor_error(sc, res.camera, rng), 95) < 0.05


def test_homography_without_distortion_is_bad_on_fisheye():
    """Защита от регресса: на сыром fisheye чистая H даёт метровые ошибки."""
    sc = SynthCam()
    cam = Camera("synth", sc.size, distortion=None)
    uv, XY = _true_floor_pairs(sc)
    hf = fit_homography(cam, uv, XY, ransac_thresh_m=10.0)
    assert hf.err_m.max() > 0.3


def test_project_unproject_roundtrip_and_horizon():
    sc = SynthCam()
    cam = Camera("synth", sc.size, distortion=sc.model)
    uv, XY = _true_floor_pairs(sc)
    cam.H = fit_homography(cam, uv, XY).H
    back, ok = cam.unproject(XY)
    assert ok.all() and np.abs(back - uv).max() < 1e-4
    # Пиксель у верхнего края кадра смотрит выше горизонта (камера наклонена на 45°,
    # горизонт на 45° от оси) — должен быть invalid, а не точка в километре.
    p, ok = cam.project([[sc.model.cx, 5.0]])
    assert not ok[0] and np.isnan(p).all()


def test_meters_per_pixel_grows_with_distance():
    sc = SynthCam()
    cam = Camera("synth", sc.size, distortion=sc.model)
    uv, XY = _true_floor_pairs(sc)
    cam.H = fit_homography(cam, uv, XY).H
    near, _ = sc.floor_to_pixels([[5, 2.0]])
    far, _ = sc.floor_to_pixels([[5, 7.0]])
    assert cam.meters_per_pixel(far)[0] > 2 * cam.meters_per_pixel(near)[0]


def test_ransac_rejects_misclick():
    sc = SynthCam()
    cam = Camera("synth", sc.size, distortion=sc.model)
    uv, XY = _true_floor_pairs(sc, 12)
    XY = XY.copy()
    XY[3] += [1.5, -1.0]                    # одна пара кликнута не туда
    hf = fit_homography(cam, uv, XY)
    assert not hf.inliers[3] and hf.inliers.sum() == 11
    assert hf.summary()["mean"] < 0.01


def test_polygons():
    sq = [[0, 0], [4, 0], [4, 4], [0, 4]]
    pts = [[2, 2], [5, 2], [4.3, 2], [-1, -1]]
    assert P.contains(sq, pts).tolist() == [True, False, False, False]
    assert P.contains_tol(sq, pts, 0.5).tolist() == [True, False, True, False]
    assert np.allclose(P.signed_distance(sq, pts), [2, -1, -0.3, -np.sqrt(2)])
    assert P.area(sq) == 16


def test_scale_from_segments():
    s = scale_from_segments([[(0, 0), (51, 0)], [(10, 10), (10, 61)], [(0, 0), (52, 0)]], [1.2, 1.2, 1.2])
    assert abs(s["m_per_px"] - 1.2 / 51) < 1e-9 and s["n"] == 3
