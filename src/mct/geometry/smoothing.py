"""Сглаживание траектории одного треклета: медиана → Калман (постоянная скорость) → RTS.

Офлайн, поэтому используем и будущие точки (RTS): нет запаздывания на поворотах.
Переменный шаг по времени: дыры внутри треклета (после сшивания У4) проходятся предсказанием.
Выбросы (расстояние Махаланобиса > порога) не обновляют фильтр и помечаются.
"""
from __future__ import annotations

import numpy as np


def rolling_median(xy: np.ndarray, win: int = 5) -> np.ndarray:
    if win <= 1 or len(xy) < 3:
        return xy.copy()
    h = win // 2
    pad = np.pad(xy, ((h, h), (0, 0)), mode="edge")
    idx = np.arange(len(xy))[:, None] + np.arange(win)[None]
    return np.median(pad[idx], axis=1)


def kalman_rts(t, z, r, q: float = 0.5, gate: float = 3.5, v0_sigma: float = 1.5):
    """t (N,), z (N, 2) метры, r (N,) сигма измерения, м.
    -> x (N, 2), v (N, 2), sigma (N,) — СКО положения после сглаживания, outlier (N,)."""
    n = len(t)
    if n == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0), np.zeros(0, bool)
    H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    xf = np.zeros((n, 4)); Pf = np.zeros((n, 4, 4))
    xp = np.zeros((n, 4)); Pp = np.zeros((n, 4, 4)); F_all = np.zeros((n, 4, 4))
    out = np.zeros(n, bool)
    x = np.r_[z[0], 0.0, 0.0]
    P = np.diag([r[0] ** 2, r[0] ** 2, v0_sigma ** 2, v0_sigma ** 2])
    for k in range(n):
        dt = 0.0 if k == 0 else max(float(t[k] - t[k - 1]), 1e-3)
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1.0]])
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        Q = (q ** 2) * G @ G.T
        if k > 0:
            x = F @ x
            P = F @ P @ F.T + Q
        F_all[k] = F; xp[k] = x; Pp[k] = P
        R = np.eye(2) * r[k] ** 2
        S = H @ P @ H.T + R
        y = z[k] - H @ x
        d2 = float(y @ np.linalg.solve(S, y))
        if k > 0 and d2 > gate ** 2:
            out[k] = True                         # выброс: только предсказание
        else:
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(4) - K @ H) @ P
        xf[k] = x; Pf[k] = P
    xs = xf.copy(); Ps = Pf.copy()
    for k in range(n - 2, -1, -1):
        C = Pf[k] @ F_all[k + 1].T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
    sig = np.sqrt(np.maximum((Ps[:, 0, 0] + Ps[:, 1, 1]) / 2, 0))
    return xs[:, :2], xs[:, 2:], sig, out
