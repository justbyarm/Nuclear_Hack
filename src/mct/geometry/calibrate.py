"""Совместная калибровка одной камеры: дисторсия fisheye + положение камеры над полом.

Почему не «отдельно дисторсия, потом H»: по одним прямым f и k частично
взаимозаменяемы, и на краях кадра без линий модель врёт на полметра.
Пары «кадр ↔ план» несут метрику и снимают эту неоднозначность.

Физическая модель (11 параметров) вместо H (8) + дисторсии (5):
  - меньше свободы → устойчивее при 10–14 парах;
  - даёт высоту подвеса камеры — проверка здравого смысла (ожидаем 2.5–3.5 м);
  - плоскость пола как гомография получается из неё точно: H хранится в json
    как раньше, остальной код ничего не замечает.

Система координат плана: x вправо, y вниз (как картинка), z = x × y — в пол.
Камера над полом, значит её центр имеет z < 0; высота = −C_z.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares

from .camera import Camera, Fisheye, normalize_H_sign
from .distortion import _plane_residuals, fit_fisheye_from_lines
from .fit import fit_homography


@dataclass
class CalibResult:
    camera: Camera
    height_m: float
    tilt_deg: float                 # угол оси объектива от вертикали
    line_rms_px: float | None
    pair_err_m: np.ndarray          # ошибка каждой пары на плане (инлайеры), м
    pair_err_px: np.ndarray
    loo_err_m: np.ndarray
    inliers: np.ndarray
    notes: list = field(default_factory=list)

    def summary(self) -> dict:
        inl = self.inliers
        loo = self.loo_err_m[inl & np.isfinite(self.loo_err_m)]
        return {
            "height_m": round(self.height_m, 3), "tilt_deg": round(self.tilt_deg, 1),
            "line_rms_px": None if self.line_rms_px is None else round(self.line_rms_px, 2),
            "n_pairs": int(len(inl)), "n_inliers": int(inl.sum()),
            "pair_mean_m": round(float(self.pair_err_m[inl].mean()), 3),
            "pair_max_m": round(float(self.pair_err_m[inl].max()), 3),
            "loo_mean_m": round(float(loo.mean()), 3) if len(loo) else None,
            "loo_max_m": round(float(loo.max()), 3) if len(loo) else None,
            "notes": self.notes,
        }


# ------------------------------------------------------------ поза <-> H

def pose_to_H(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Идеальные координаты (x, y, 1) -> план (X, Y). Пол: Z = 0."""
    G = np.column_stack([R[:, 0], R[:, 1], t])        # план -> идеальные
    return np.linalg.inv(G)


def H_to_pose(H: np.ndarray, ideal_pts):
    """Разложение гомографии пола в позу камеры (R, t), камера над полом."""
    G = np.linalg.inv(H)
    lam = 2.0 / (np.linalg.norm(G[:, 0]) + np.linalg.norm(G[:, 1]))
    best = None
    for s in (1, -1):
        r1, r2, t = s * lam * G[:, 0], s * lam * G[:, 1], s * lam * G[:, 2]
        U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = U @ np.diag([1, 1, -1]) @ Vt
        # Точки пола должны быть перед камерой.
        XY = _apply(H, ideal_pts)
        depth = (np.column_stack([XY, np.zeros(len(XY))]) @ R.T + t)[:, 2]
        score = (depth > 0).mean()
        if best is None or score > best[0]:
            best = (score, R, t)
    return best[1], best[2]


def _apply(H, xy):
    ph = np.column_stack([xy, np.ones(len(xy))]) @ H.T
    return ph[:, :2] / ph[:, 2:3]


def floor_to_pixels(model: Fisheye, R, t, XY):
    XY = np.asarray(XY, dtype=np.float64).reshape(-1, 2)
    c = np.column_stack([XY, np.zeros(len(XY))]) @ R.T + t
    b = c / np.linalg.norm(c, axis=1, keepdims=True)
    return model.bearings_to_pixels(b), c[:, 2] > 0


# ------------------------------------------------------------ калибровка

def calibrate_camera(camera_id: str, image_size, img_pts, plan_pts_m, lines=(),
                     n_k: int = 2, ransac_thresh_m: float = 0.3,
                     line_weight: float = 1.0) -> CalibResult:
    """img_pts — пиксели исходного кадра; plan_pts_m — метры плана; lines — ломаные вдоль прямых."""
    img = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
    plan = np.asarray(plan_pts_m, dtype=np.float64).reshape(-1, 2)
    lines = [np.asarray(l, dtype=np.float64).reshape(-1, 2) for l in lines if len(l) >= 3]
    notes = []
    if len(img) < 6:
        raise ValueError("для совместной калибровки нужно хотя бы 6 пар, лучше 10–14")

    W, Hh = image_size
    if len(lines) < 3:
        notes.append("нет прямых: дисторсия только по парам, нужно больше пар по краям кадра")

    def unpack(p):
        f, cx, cy = p[0], p[1], p[2]
        k = tuple(p[3:3 + n_k]) + (0.0,) * (4 - n_k)
        rvec, t = p[3 + n_k:6 + n_k], p[6 + n_k:9 + n_k]
        return Fisheye(f, cx, cy, k), cv2.Rodrigues(rvec)[0], t

    def resid(p, use):
        m, R, t = unpack(p)
        uv, front = floor_to_pixels(m, R, t, plan[use])
        r = [np.where(front[:, None], uv - img[use], 1e3).ravel()]
        if lines:
            r.append(line_weight * np.concatenate([_plane_residuals(m.pixels_to_bearings(l)) for l in lines]) * m.f)
        r.append([(m.cx - W / 2) / (0.05 * W), (m.cy - Hh / 2) / (0.05 * Hh)])
        return np.concatenate(r)

    f_ref = (W / 2) / np.deg2rad(60)
    lo = np.r_[0.15 * f_ref, 0.3 * W, 0.3 * Hh, -np.ones(n_k), -np.full(6, np.inf)]
    hi = np.r_[6 * f_ref, 0.7 * W, 0.7 * Hh, np.ones(n_k), np.full(6, np.inf)]

    # 1–3. Мультистарт: начальная дисторсия (по прямым или «полуширина кадра = A°»),
    # из неё H по парам (RANSAC) -> поза -> совместный подбор. Берём лучший старт.
    starts = [Fisheye((W / 2) / np.deg2rad(a), W / 2, Hh / 2) for a in (40, 55, 70, 85, 100)]
    if len(lines) >= 3:
        starts.insert(0, fit_fisheye_from_lines(lines, image_size, n_k=n_k).model)
    best = None
    for init in starts:
        cam0 = Camera(camera_id, tuple(image_size), distortion=init)
        try:
            hf = fit_homography(cam0, img, plan, ransac_thresh_m=max(2 * ransac_thresh_m, 0.6))
        except (ValueError, RuntimeError):
            continue
        inl0 = hf.inliers.copy()
        if inl0.sum() < 6:
            continue
        ideal0, _ = cam0.to_ideal(img)
        R0, t0 = H_to_pose(hf.H, ideal0[inl0])
        p0 = np.clip(np.r_[init.f, init.cx, init.cy, init.k[:n_k], cv2.Rodrigues(R0)[0].ravel(), t0],
                     lo + 1e-9, hi - 1e-9)
        try:
            # Сначала без робастной потери — уходим из плохого старта,
            # потом soft_l1 — чтобы единичный промах клика не тянул решение.
            p1 = least_squares(resid, p0, args=(inl0,), bounds=(lo, hi), x_scale="jac").x
            p1 = least_squares(resid, p1, args=(inl0,), bounds=(lo, hi),
                               loss="soft_l1", f_scale=3.0, x_scale="jac")
        except ValueError:
            continue
        pair_px = np.abs(p1.fun[:2 * inl0.sum()])
        score = float(np.median(pair_px)) + 0.1 * (~inl0).sum()
        if best is None or score < best[0]:
            best = (score, p1.x, inl0)
    if best is None:
        raise RuntimeError("калибровка не сошлась ни с одного старта: проверь пары точек")
    _, p, inl = best

    # 4. Повторный отсев промахов по ошибке на плане и финальная подгонка.
    m, R, t = unpack(p)
    H = pose_to_H(R, t)
    cam = Camera(camera_id, tuple(image_size), H=None, distortion=m)
    ideal, _ = cam.to_ideal(img)
    H = normalize_H_sign(H, ideal[inl])
    err_m = np.linalg.norm(_apply(H, ideal) - plan, axis=1)
    new_inl = err_m < ransac_thresh_m
    if new_inl.sum() >= 6 and (new_inl != inl).any():
        inl = new_inl
        p = least_squares(resid, p, args=(inl,), bounds=(lo, hi),
                          loss="soft_l1", f_scale=3.0, x_scale="jac").x
        m, R, t = unpack(p)
        cam.distortion = m
        ideal, _ = cam.to_ideal(img)
        H = normalize_H_sign(pose_to_H(R, t), ideal[inl])
        err_m = np.linalg.norm(_apply(H, ideal) - plan, axis=1)
    cam.H = H
    uv, _ = floor_to_pixels(m, R, t, plan)
    err_px = np.linalg.norm(uv - img, axis=1)

    # 5. Leave-one-out: интринсики фиксируем, позу переподбираем без точки i.
    loo = np.full(len(img), np.nan)
    idx = np.where(inl)[0]
    if len(idx) >= 7:
        rv = cv2.Rodrigues(R)[0].ravel()
        for i in idx:
            rest = idx[idx != i]

            def rpose(q):
                Ri = cv2.Rodrigues(q[:3])[0]
                u, _ = floor_to_pixels(m, Ri, q[3:], plan[rest])
                return (u - img[rest]).ravel()
            q = least_squares(rpose, np.r_[rv, t], loss="soft_l1", f_scale=3.0).x
            Hi = pose_to_H(cv2.Rodrigues(q[:3])[0], q[3:])
            loo[i] = np.linalg.norm(_apply(Hi, ideal[i:i + 1]) - plan[i])

    C = -R.T @ t
    # Знак C_z зависит от того, смотрит ли ось y плана «вниз» (картинка) или «вверх»;
    # высота — модуль.
    height = float(abs(C[2]))
    axis_world = R.T @ np.array([0, 0, 1.0])
    # 0° = смотрит вертикально в пол, 90° = горизонтально.
    tilt = float(np.degrees(np.arccos(np.clip(abs(axis_world[2]), 0, 1))))
    if not 1.8 < height < 5.0:
        notes.append(f"высота камеры {height:.2f} м подозрительна: проверь масштаб плана и пары")
    line_rms = None
    if lines:
        lr = np.concatenate([_plane_residuals(m.pixels_to_bearings(l)) for l in lines]) * m.f
        line_rms = float(np.sqrt(np.mean(lr ** 2)))

    cam.raw.update({"pose": {"rvec": cv2.Rodrigues(R)[0].ravel().tolist(), "t": t.tolist(),
                             "center_m": C.tolist(), "height_m": height}})
    return CalibResult(cam, height, tilt, line_rms, err_m, err_px, loo, inl, notes)
