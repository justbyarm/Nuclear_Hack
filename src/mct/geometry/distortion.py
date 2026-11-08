"""Оценка дисторсии fisheye по прямым (plumb-line), без шахматной доски.

Вход: несколько ломаных, кликнутых вдоль линий, прямых в реальном мире
(швы плитки, края стен у пола, дверные косяки), по 5–15 точек на линию.

Идея: прямая в 3D вместе с центром камеры лежит в плоскости, поэтому лучи
всех её точек ортогональны нормали этой плоскости. Подбираем f, cx, cy, k1, k2,
при которых для каждой линии лучи лучше всего лежат в одной плоскости.
Невязка в радианах — работает и на краю кадра, где пинхол-координаты взрываются.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .camera import Fisheye


def _plane_residuals(b: np.ndarray) -> np.ndarray:
    """Отклонение лучей от наилучшей плоскости через центр камеры (радианы)."""
    _, _, vt = np.linalg.svd(b, full_matrices=False)
    n = vt[-1]
    return np.arcsin(np.clip(b @ n, -1, 1))


def line_residuals(model: Fisheye, lines) -> np.ndarray:
    return np.concatenate([_plane_residuals(model.pixels_to_bearings(l)) for l in lines])


@dataclass
class DistortionFit:
    model: Fisheye
    rms_before_px: float     # RMS отклонения от прямой, если считать камеру пинхолом
    rms_after_px: float      # то же после подбора fisheye
    per_line_px: list        # RMS по каждой линии после подбора
    n_lines: int
    n_points: int


def fit_fisheye_from_lines(lines, image_size, f0: float | None = None,
                           fit_center: bool = True, n_k: int = 2,
                           k_weight: float = 0.0) -> DistortionFit:
    """lines: список массивов (Ni, 2) пикселей, Ni >= 3, лучше >= 6 линий по всему кадру."""
    lines = [np.asarray(l, dtype=np.float64).reshape(-1, 2) for l in lines if len(l) >= 3]
    if len(lines) < 3:
        raise ValueError("нужно хотя бы 3 линии по 3+ точки, лучше 6–10 по всему кадру")
    W, H = image_size
    # Начальное f: ~ полуширина кадра соответствует ~60° от оси.
    f0 = f0 or (W / 2) / np.deg2rad(60)
    cx0, cy0 = W / 2, H / 2
    n_k = int(np.clip(n_k, 1, 4))

    def unpack(p, with_center):
        f = p[0]
        if with_center:
            cx, cy, ks = p[1], p[2], p[3:]
        else:
            cx, cy, ks = cx0, cy0, p[1:]
        k = tuple(ks) + (0.0,) * (4 - len(ks))
        return Fisheye(f, cx, cy, k)

    def resid(p, with_center):
        m = unpack(p, with_center)
        # Невязка в пикселях: радианы × текущий f. С фиксированным множителем
        # оптимизатор «сжимает» лучи к оси (f → ∞) и выигрывает впустую.
        r = line_residuals(m, lines) * m.f
        # f и k частично взаимозаменяемы: по одним прямым края кадра без линий
        # экстраполируются плохо. Это лечит совместная калибровка с парами
        # (calibrate.py); штраф на k оставлен опцией, по умолчанию выключен.
        if k_weight:
            r = np.concatenate([r, k_weight * np.asarray(m.k[:n_k]) / 0.05])
        # Регуляризация центра: не дальше ~5% кадра без причины.
        if with_center:
            r = np.concatenate([r, [(m.cx - cx0) / (0.05 * W), (m.cy - cy0) / (0.05 * H)]])
        return r

    # Шаг 1: только f и k при центре в середине кадра.
    p1 = least_squares(resid, np.r_[f0, np.zeros(n_k)], args=(False,),
                       bounds=(np.r_[0.2 * f0, -np.ones(n_k)], np.r_[5 * f0, np.ones(n_k)]),
                       loss="soft_l1", f_scale=2.0).x
    best = unpack(p1, False)
    # Шаг 2: отпускаем центр.
    if fit_center:
        p2 = least_squares(resid, np.r_[p1[0], cx0, cy0, p1[1:]], args=(True,),
                           bounds=(np.r_[0.2 * f0, 0.3 * W, 0.3 * H, -np.ones(n_k)],
                                   np.r_[5 * f0, 0.7 * W, 0.7 * H, np.ones(n_k)]),
                           loss="soft_l1", f_scale=2.0).x
        best = unpack(p2, True)

    after = [float(np.sqrt(np.mean(_plane_residuals(best.pixels_to_bearings(l)) ** 2)) * best.f)
             for l in lines]
    allr = line_residuals(best, lines) * best.f
    return DistortionFit(
        model=best,
        rms_before_px=_pinhole_straightness_rms(lines),
        rms_after_px=float(np.sqrt(np.mean(allr ** 2))),
        per_line_px=after,
        n_lines=len(lines),
        n_points=int(sum(len(l) for l in lines)),
    )


def _pinhole_straightness_rms(lines) -> float:
    """RMS расстояния точек от прямой, подогнанной в сырых пикселях."""
    res = []
    for l in lines:
        c = l - l.mean(0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        res.append(c @ vt[-1])
    r = np.concatenate(res)
    return float(np.sqrt(np.mean(r ** 2)))


def undistort_image(img: np.ndarray, model: Fisheye, out_size=None, f_out: float | None = None,
                    max_angle_deg: float = 80.0) -> np.ndarray:
    """Распрямлённая (пинхол) картинка — для глаз и отладки, не для расчётов."""
    import cv2
    H, W = img.shape[:2]
    Wo, Ho = out_size or (W, H)
    # Фокус выхода: чтобы угол max_angle_deg попал в край по ширине.
    f_out = f_out or (Wo / 2) / np.tan(np.deg2rad(max_angle_deg))
    xs, ys = np.meshgrid(np.arange(Wo), np.arange(Ho))
    xy = np.stack([(xs - Wo / 2) / f_out, (ys - Ho / 2) / f_out], -1).reshape(-1, 2)
    uv = model.distort(xy).reshape(Ho, Wo, 2).astype(np.float32)
    return cv2.remap(img, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
