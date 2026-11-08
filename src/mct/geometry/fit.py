"""Подбор гомографии по парам точек «кадр ↔ план» и оценка ошибки."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import Camera, normalize_H_sign


@dataclass
class HomographyFit:
    H: np.ndarray
    inliers: np.ndarray        # bool (N,)
    err_m: np.ndarray          # ошибка каждой пары на плане, м
    loo_err_m: np.ndarray      # ошибка leave-one-out, м (NaN, если не посчитать)

    def summary(self) -> dict:
        inl = self.inliers
        loo = self.loo_err_m[inl & np.isfinite(self.loo_err_m)]
        return {
            "n_pairs": int(len(inl)), "n_inliers": int(inl.sum()),
            "mean": float(self.err_m[inl].mean()), "max": float(self.err_m[inl].max()),
            "loo_mean": float(loo.mean()) if len(loo) else None,
            "loo_max": float(loo.max()) if len(loo) else None,
        }


def _apply(H, xy):
    ph = np.column_stack([xy, np.ones(len(xy))]) @ H.T
    return ph[:, :2] / ph[:, 2:3]


def fit_homography(cam: Camera, img_pts, plan_pts_m, ransac_thresh_m: float = 0.3) -> HomographyFit:
    """img_pts — пиксели исходного кадра, plan_pts_m — метры плана. Нужно >= 4 пар, лучше 10+."""
    img_pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
    plan = np.asarray(plan_pts_m, dtype=np.float64).reshape(-1, 2)
    if len(img_pts) != len(plan):
        raise ValueError("число точек на кадре и на плане не совпадает")
    if len(img_pts) < 4:
        raise ValueError("нужно хотя бы 4 пары, лучше 10–14 по всему полу")
    ideal, ok = cam.to_ideal(img_pts)
    if not ok.all():
        raise ValueError(f"точки {np.where(~ok)[0].tolist()} за 90° от оси объектива — проверь дисторсию")

    method = cv2.RANSAC if len(img_pts) >= 6 else 0
    H, mask = cv2.findHomography(ideal, plan, method, ransac_thresh_m)
    if H is None:
        raise RuntimeError("гомография не нашлась: точки почти на одной прямой?")
    inl = mask.ravel().astype(bool) if mask is not None else np.ones(len(ideal), bool)
    # Уточняем по всем инлайерам МНК.
    if inl.sum() >= 4:
        H, _ = cv2.findHomography(ideal[inl], plan[inl], 0)
    H = normalize_H_sign(H, ideal[inl])
    err = np.linalg.norm(_apply(H, ideal) - plan, axis=1)

    loo = np.full(len(ideal), np.nan)
    idx = np.where(inl)[0]
    if len(idx) >= 6:
        for i in idx:
            rest = idx[idx != i]
            Hi, _ = cv2.findHomography(ideal[rest], plan[rest], 0)
            if Hi is not None:
                loo[i] = np.linalg.norm(_apply(Hi, ideal[i:i + 1]) - plan[i])
    return HomographyFit(H, inl, err, loo)
