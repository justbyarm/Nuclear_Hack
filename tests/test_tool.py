"""Сквозная проверка инструмента без окон: клики подаём из кода."""
import json

import cv2
import numpy as np

from mct.geometry.camera import Camera
from mct.geometry.plan import Plan
from mct.geometry.tool import CamSession, PlanSession, View

from synthetic import SynthCam

PX_PER_M = 50          # синтетический план: 50 px на метр
TILE = 0.6             # плитка 0.6 × 0.6 м


def render_floor(sc: SynthCam) -> np.ndarray:
    """Кадр синтетической камеры: плитка на полу, стены — серые."""
    W, H = sc.size
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    b = sc.model.pixels_to_bearings(np.column_stack([u.ravel(), v.ravel()]))
    d = b @ sc.R                                       # камера -> мир
    s = -sc.C[2] / np.where(d[:, 2] < -1e-6, d[:, 2], np.nan)
    X, Y = sc.C[0] + s * d[:, 0], sc.C[1] + s * d[:, 1]
    on = np.isfinite(s) & (X > 0) & (X < 10) & (Y > 0) & (Y < 10)
    chk = ((np.floor(X / TILE) + np.floor(Y / TILE)) % 2 == 0)
    img = np.full((H * W, 3), 90, np.uint8)
    img[on & chk] = (200, 200, 200)
    img[on & ~chk] = (150, 150, 150)
    return img.reshape(H, W, 3)


def render_plan() -> np.ndarray:
    img = np.full((10 * PX_PER_M, 10 * PX_PER_M, 3), 255, np.uint8)
    for k in np.arange(0, 10 + 1e-9, TILE):
        p = int(round(k * PX_PER_M))
        cv2.line(img, (p, 0), (p, img.shape[0]), (180, 180, 180), 1)
        cv2.line(img, (0, p), (img.shape[1], p), (180, 180, 180), 1)
    return img


def make_plan(tmp_path):
    ps = PlanSession(render_plan(), tmp_path / "plan.json", "plan.png", seg_m=1.2)
    ps.set_mode("k")
    for y in (20, 200, 420):                            # три отрезка по 1.2 м = 60 px
        ps.click(100, y); ps.click(160, y)
    ps.set_mode("f")
    for p in [(0, 0), (500, 0), (500, 500), (0, 500)]:
        ps.click(*p)
    ps.finish()
    ps.set_mode("x")
    for p in [(0, 200), (30, 200), (30, 260), (0, 260)]:
        ps.click(*p)
    ps.finish(); ps.set_exit_type(ord("3"))
    plan = ps.save()
    assert abs(plan.m_per_px - 1 / PX_PER_M) < 1e-12
    assert plan.exits[0]["type"] == "entrance"
    assert Plan.load(tmp_path / "plan.json").on_floor([[5, 5], [11, 5]], 0.3).tolist() == [True, False]
    return plan


def test_plan_and_camera_sessions_end_to_end(tmp_path):
    rng = np.random.default_rng(3)
    sc = SynthCam()
    frame, plan_img = render_floor(sc), render_plan()
    plan = make_plan(tmp_path)

    s = CamSession("cam_s", frame, plan_img, plan, tmp_path / "cam_s.json")
    # Прямые: швы плитки по всему кадру, с дрожанием руки.
    s.set_mode("l")
    for line in sc.floor_lines(xs=np.arange(0.6, 9.7, 1.2), ys=np.arange(1.2, 9.7, 1.2)):
        for p in line + rng.normal(0, 0.7, line.shape):
            s.click_frame(*p)
        s.finish()
    # Пары: углы плиток.
    s.set_mode("p")
    corners = np.array([[x, y] for x in np.arange(1.2, 9, 1.2) for y in np.arange(1.2, 9, 1.2)])
    uv, vis = sc.floor_to_pixels(corners)
    sel = rng.choice(np.where(vis)[0], 12, replace=False)
    for i in sel:
        s.click_frame(*(uv[i] + rng.normal(0, 1, 2)))
        s.click_plan(*(corners[i] * PX_PER_M + rng.normal(0, 0.5, 2)))
    # Видимый пол и стол.
    s.set_mode("f")
    for p in [(200, 1000), (2400, 1000), (2500, 1900), (100, 1900)]:
        s.click_frame(*p)
    s.finish()
    s.set_mode("d")
    for p in [(1100, 1300), (1500, 1300), (1500, 1500), (1100, 1500)]:
        s.click_frame(*p)
    s.finish()

    res = s.calibrate()
    assert res is not None, s.message
    summ = res.summary()
    assert summ["n_inliers"] == 12 and summ["loo_mean_m"] < 0.06, summ
    assert abs(res.height_m - 2.8) < 0.1

    fr, pl = s.render_frame(), s.render_plan()
    assert fr.shape == frame.shape and pl.shape == plan_img.shape
    cv2.imwrite(str(tmp_path / "frame_grid.jpg"), fr)

    path = s.save()
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["image_size"] == [2560, 1920] and len(d["fov_plan"]) > 20
    cam = Camera.load(path)
    p = cam.project_one(*uv[sel[0]])
    assert np.hypot(p[0] - corners[sel[0]][0], p[1] - corners[sel[0]][1]) < 0.08

    # Повторное открытие подхватывает все клики.
    s2 = CamSession("cam_s", frame, plan_img, plan, path)
    assert len(s2.lines) == len(s.lines) and len(s2.complete_pairs()) == 12 and len(s2.desks) == 1


def test_undo_and_pair_order():
    sc = SynthCam()
    s = CamSession("c", np.zeros((1920, 2560, 3), np.uint8), np.zeros((500, 500, 3), np.uint8), None, "/nonexistent/c.json")
    s.set_mode("p")
    s.click_plan(10, 10)
    assert "сначала" in s.message and not s.pairs
    s.click_frame(100, 100); s.click_frame(120, 120)          # перекликнули точку кадра
    assert len(s.pairs) == 1 and s.pairs[0]["img"] == [120, 120]
    s.click_plan(10, 10); s.undo()
    assert s.pairs[0]["plan_px"] is None
    s.undo()
    assert not s.pairs
    assert s.calibrate() is None and "plan.json" in s.message


def test_view_zoom_keeps_point_under_cursor():
    v = View(2560, 1920, 1280, 900)
    u0 = v.to_img(300, 200)
    v.zoom_at(300, 200, 3.0)
    assert np.allclose(v.to_img(300, 200), u0)
    v.pan(50, 0)
    assert np.allclose(v.to_win(*v.to_img(10, 10)), (10, 10))
