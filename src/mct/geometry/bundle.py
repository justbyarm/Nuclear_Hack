"""Совместная калибровка нескольких камер (bundle adjustment) по разнородным наблюдениям.

Зачем: мебель на плане устарела, точных точек «кадр ↔ план» в холле мало. Зато есть
  - стены плана: точка низа стены на кадре должна лечь на линию стены на плане;
  - предметы известного размера (теннисный стол 2,74 × 1,525 м, высота 0,76 м):
    задают масштаб и связывают камеры, которые видят один предмет;
  - общие точки на полу (ножка стола, угол стойки), видимые двумя камерами:
    их положение неизвестно, но оно одно для всех камер → камеры согласованы между собой;
  - прямые (швы плитки, низ стен, рейки, косяки) → дисторсия.

Мир: план в метрах, x вправо, y вниз, z = x × y смотрит В ПОЛ. Высота h над полом → Z = −h.

Наблюдения одной камеры (json, пиксели исходного кадра, план — в пикселях картинки плана):
{
  "camera_id": "cam1", "image_size": [2592, 1944],
  "hint": {"center_px": [x, y], "look_at_px": [x, y], "height_m": 2.8},
  "lines":  [[[u, v], ...], ...],                         # прямые в 3D (любые)
  "verticals": [[[u, v], ...], ...],                      # прямые, вертикальные в мире
  "pairs":  [{"img": [u, v], "plan_px": [x, y], "label": "..."}],
  "walls":  [{"img": [[u, v], ...], "plan_px": [[x1, y1], [x2, y2]], "label": "..."}],
  "objects": [{"id": "pingpong", "img": [[u, v] x 4]}],     # углы по кругу
  "landmarks": [{"id": "stand_leg", "img": [u, v]}]
}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from .camera import Camera, Fisheye, normalize_H_sign
from .distortion import _plane_residuals

OBJECTS = {"pingpong": {"size_m": (2.74, 1.525), "height_m": 0.76}}
N_K = 2
# Реальные fisheye: |k1| ~ 0.01–0.1. Широкие границы дают немонотонную модель
# на краю кадра, и оптимизатор уходит в мусор.
K_BOUND = np.array([0.3, 0.1, 0.05, 0.02])
MISS = 50.0     # невязка (px), если луч не попал на пол или точка позади камеры


# ------------------------------------------------------------------ геометрия

def look_at_R(C, target):
    fwd = np.asarray(target, float) - np.asarray(C, float)
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, -1.0])
    x = np.cross(fwd, up)
    x /= np.linalg.norm(x)
    y = np.cross(fwd, x)
    return np.stack([x, y, fwd])


def project(model: Fisheye, R, C, P):
    """Мировые точки (N, 3) -> пиксели (N, 2), front (N,) — точка перед камерой."""
    c = (np.asarray(P, float).reshape(-1, 3) - C) @ R.T
    n = np.linalg.norm(c, axis=1, keepdims=True)
    return model.bearings_to_pixels(c / np.maximum(n, 1e-12)), c[:, 2] > 1e-6


def backproject_floor(model: Fisheye, R, C, uv, Z=0.0):
    """Пиксели -> точки на плоскости Z (пол Z=0), ok — луч идёт к полу."""
    d = model.pixels_to_bearings(uv) @ R          # камера -> мир
    s = (Z - C[2]) / np.where(np.abs(d[:, 2]) < 1e-9, 1e-9, d[:, 2])
    P = C + s[:, None] * d
    return P[:, :2], s > 0


def object_corners(obj_def, p):
    """p = (ox, oy, theta) -> 4 угла (4, 3). Угол 0→1 — длинная сторона."""
    L, W = obj_def["size_m"]
    h = obj_def["height_m"]
    loc = np.array([[-L / 2, -W / 2], [L / 2, -W / 2], [L / 2, W / 2], [-L / 2, W / 2]])
    c, s = np.cos(p[2]), np.sin(p[2])
    xy = loc @ np.array([[c, s], [-s, c]]) + p[:2]
    return np.column_stack([xy, np.full(4, -h)])


# ------------------------------------------------------------------ задача

@dataclass
class CamObs:
    camera_id: str
    image_size: tuple
    hint: dict
    lines: list = field(default_factory=list)
    verticals: list = field(default_factory=list)
    pairs: list = field(default_factory=list)          # (img (2,), plan_m (2,))
    walls: list = field(default_factory=list)          # (img (N,2), a (2,), b (2,))
    objects: list = field(default_factory=list)        # (id, img (4,2))
    landmarks: list = field(default_factory=list)      # (id, img (2,))

    @classmethod
    def from_json(cls, d: dict, m_per_px: float) -> "CamObs":
        m = lambda p: np.asarray(p, float) * m_per_px
        return cls(
            d["camera_id"], tuple(d["image_size"]), d.get("hint", {}),
            [np.asarray(l, float) for l in d.get("lines", []) if len(l) >= 3],
            [np.asarray(l, float) for l in d.get("verticals", []) if len(l) >= 2],
            [(np.asarray(p["img"], float), m(p["plan_px"])) for p in d.get("pairs", [])],
            [(np.asarray(w["img"], float), *m(w["plan_px"])) for w in d.get("walls", [])],
            [(o["id"], np.asarray(o["img"], float)) for o in d.get("objects", [])],
            [(l["id"], np.asarray(l["img"], float)) for l in d.get("landmarks", [])],
        )


class Bundle:
    """Параметры: на камеру f, cx, cy, k1, k2, rvec(3), C(3); на объект ox, oy, θ; на общую точку X, Y."""

    def __init__(self, cams: list[CamObs], m_per_px: float, wall_w: float = 1.0,
                 line_w: float = 1.0, height_prior=(2.8, 1.5)):
        self.cams, self.mpp = cams, m_per_px
        self.wall_w, self.line_w, self.height_prior = wall_w, line_w, height_prior
        self.obj_ids = sorted({o for c in cams for o, _ in c.objects})
        lm_count = {}
        for c in cams:
            for i, _ in c.landmarks:
                lm_count[i] = lm_count.get(i, 0) + 1
        # Точка, которую видит одна камера, ничего не связывает — её не берём.
        self.lm_ids = sorted(i for i, n in lm_count.items() if n >= 2)
        self.nc = 3 + N_K + 6      # f cx cy k1 k2 rvec(3) C(3) = 11

    # --- упаковка
    def n_params(self):
        return self.nc * len(self.cams) + 3 * len(self.obj_ids) + 2 * len(self.lm_ids)

    def cam_slice(self, i):
        return slice(i * self.nc, (i + 1) * self.nc)

    def obj_slice(self, j):
        o = self.nc * len(self.cams)
        return slice(o + 3 * j, o + 3 * j + 3)

    def lm_slice(self, j):
        o = self.nc * len(self.cams) + 3 * len(self.obj_ids)
        return slice(o + 2 * j, o + 2 * j + 2)

    def unpack_cam(self, q):
        f, cx, cy = q[0], q[1], q[2]
        k = tuple(q[3:3 + N_K]) + (0.0,) * (4 - N_K)
        R = cv2.Rodrigues(np.asarray(q[3 + N_K:6 + N_K], float))[0]
        return Fisheye(f, cx, cy, k), R, np.asarray(q[6 + N_K:9 + N_K], float)

    def pack_cam(self, model: Fisheye, R, C):
        return np.r_[model.f, model.cx, model.cy, model.k[:N_K], cv2.Rodrigues(R)[0].ravel(), C]

    # --- невязки
    def residuals(self, p):
        return np.concatenate([self.cam_residuals(i, p) for i in range(len(self.cams))])

    def cam_residuals(self, i, p, detail=False):
        c = self.cams[i]
        m, R, C = self.unpack_cam(p[self.cam_slice(i)])
        W, H = c.image_size
        parts = {}
        if c.pairs:
            img = np.array([a for a, _ in c.pairs]); pl = np.array([b for _, b in c.pairs])
            uv, fr = project(m, R, C, np.column_stack([pl, np.zeros(len(pl))]))
            parts["pairs"] = np.where(fr[:, None], uv - img, MISS).ravel()
        if c.walls:
            r = []
            for img, a, b in c.walls:
                X, ok = backproject_floor(m, R, C, img)
                d = b - a
                t = ((X - a) @ d) / (d @ d)
                F = a + t[:, None] * d                    # ближайшая точка линии стены
                uv, fr = project(m, R, C, np.column_stack([F, np.zeros(len(F))]))
                r.append(np.where((ok & fr)[:, None], uv - img, MISS).ravel())
            parts["walls"] = self.wall_w * np.concatenate(r)
        if c.lines:
            parts["lines"] = self.line_w * np.concatenate(
                [_plane_residuals(m.pixels_to_bearings(l)) for l in c.lines]) * m.f
        if c.verticals:
            # Вертикаль мира в системе камеры; плоскость «центр камеры + вертикальная прямая»
            # содержит её. Сильная связь наклона камеры с дисторсией.
            vz = R[:, 2]
            r = []
            for l in c.verticals:
                bb = m.pixels_to_bearings(l)
                n = np.cross(vz, bb.mean(0))
                n /= np.linalg.norm(n) + 1e-12
                r.append(np.arcsin(np.clip(bb @ n, -1, 1)) * m.f)
            parts["verticals"] = self.line_w * np.concatenate(r)
        if c.objects:
            r = []
            for oid, img in c.objects:
                j = self.obj_ids.index(oid)
                P = object_corners(OBJECTS[oid], p[self.obj_slice(j)])
                uv, fr = project(m, R, C, P)
                r.append(np.where(fr[:, None], uv - img, MISS).ravel())
            parts["objects"] = np.concatenate(r)
        lms = [(i_, img) for i_, img in c.landmarks if i_ in self.lm_ids]
        if lms:
            P = np.array([np.r_[p[self.lm_slice(self.lm_ids.index(i_))], 0.0] for i_, _ in lms])
            uv, fr = project(m, R, C, P)
            parts["landmarks"] = np.where(fr[:, None], uv - np.array([im for _, im in lms]), MISS).ravel()
        h0, hs = self.height_prior
        parts["prior"] = np.array([(m.cx - W / 2) / (0.05 * W), (m.cy - H / 2) / (0.05 * H),
                                   (-C[2] - h0) / hs])
        if detail:
            return parts
        return np.concatenate(list(parts.values()))

    # --- старт
    def init_params(self):
        p = np.zeros(self.n_params())
        for i, c in enumerate(self.cams):
            p[self.cam_slice(i)] = self.init_cam(c)
        for j, oid in enumerate(self.obj_ids):
            p[self.obj_slice(j)] = self.init_object(oid, p)
        for j, lid in enumerate(self.lm_ids):
            pts = []
            for i, c in enumerate(self.cams):
                for k, img in c.landmarks:
                    if k == lid:
                        m, R, C = self.unpack_cam(p[self.cam_slice(i)])
                        X, ok = backproject_floor(m, R, C, img[None])
                        if ok[0]:
                            pts.append(X[0])
            p[self.lm_slice(j)] = np.mean(pts, 0) if pts else 0.0
        return p

    def init_cam(self, c: CamObs, f_deg=70.0, tilt_deg=45.0):
        W, H = c.image_size
        h = c.hint.get("height_m", 2.8)
        Cxy = np.asarray(c.hint["center_px"], float) * self.mpp
        T = np.asarray(c.hint["look_at_px"], float) * self.mpp
        dist = h / np.tan(np.deg2rad(90 - tilt_deg))      # точка пола на оси объектива
        dirv = (T - Cxy) / np.linalg.norm(T - Cxy)
        C = np.r_[Cxy, -h]
        R = look_at_R(C, np.r_[Cxy + dirv * dist, 0.0])
        model = Fisheye((W / 2) / np.deg2rad(f_deg), W / 2, H / 2)
        return self.pack_cam(model, R, C)

    def init_object(self, oid, p):
        """Положение объекта: средний по камерам обратный проект его углов на высоту объекта."""
        d = OBJECTS[oid]
        best = []
        for i, c in enumerate(self.cams):
            for k, img in c.objects:
                if k != oid:
                    continue
                m, R, C = self.unpack_cam(p[self.cam_slice(i)])
                X, ok = backproject_floor(m, R, C, img, Z=-d["height_m"])
                if ok.all():
                    ctr = X.mean(0)
                    ang = np.arctan2(*(X[1] - X[0])[::-1])
                    best.append(np.r_[ctr, ang])
        return best[0] if best else np.zeros(3)

    # --- решение
    def bounds(self):
        lo = np.full(self.n_params(), -np.inf)
        hi = np.full(self.n_params(), np.inf)
        for i, c in enumerate(self.cams):
            W, H = c.image_size
            f_ref = (W / 2) / np.deg2rad(60)
            s = self.cam_slice(i)
            lo[s.start:s.start + 3 + N_K] = np.r_[0.15 * f_ref, 0.3 * W, 0.3 * H, -K_BOUND[:N_K]]
            hi[s.start:s.start + 3 + N_K] = np.r_[6 * f_ref, 0.7 * W, 0.7 * H, K_BOUND[:N_K]]
        return lo, hi

    def sparsity(self, p):
        """Какие параметры влияют на какие невязки. Без этого численный якобиан при сотнях
        общих точек считается сотнями полных проходов; с ним — десятками."""
        from scipy.sparse import lil_matrix
        rows = []
        for i in range(len(self.cams)):
            parts = self.cam_residuals(i, p, detail=True)
            rows.append(parts)
        n = sum(len(v) for parts in rows for v in parts.values())
        S = lil_matrix((n, self.n_params()), dtype=np.int8)
        r0 = 0
        for i, parts in enumerate(rows):
            c = self.cams[i]
            cs = self.cam_slice(i)
            for name, v in parts.items():
                L = len(v)
                S[r0:r0 + L, cs] = 1
                if name == "objects":
                    k = r0
                    for oid, _ in c.objects:
                        S[k:k + 8, self.obj_slice(self.obj_ids.index(oid))] = 1
                        k += 8
                elif name == "landmarks":
                    k = r0
                    for lid, _ in c.landmarks:
                        if lid in self.lm_ids:
                            S[k:k + 2, self.lm_slice(self.lm_ids.index(lid))] = 1
                            k += 2
                r0 += L
        return S

    def solve(self, p0, robust=True, f_scale=3.0, max_nfev=None, verbose=0):
        lo, hi = self.bounds()
        p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)
        kw = {}
        if self.lm_ids or len(self.cams) > 1:
            kw["jac_sparsity"] = self.sparsity(p0)
            kw["tr_solver"] = "lsmr"
        r = least_squares(self.residuals, p0, bounds=(lo, hi), x_scale="jac",
                          loss="soft_l1" if robust else "linear", f_scale=f_scale, max_nfev=max_nfev,
                          verbose=verbose, **kw)
        return r


def calibrate_single(cam: CamObs, m_per_px: float, starts_f=(65, 95),
                     starts_tilt=(40, 60), verbose=False):
    """Одна камера: мультистарт по fov и наклону вокруг подсказки положения."""
    best = None
    for fd in starts_f:
        for td in starts_tilt:
            b = Bundle([cam], m_per_px)
            p = np.zeros(b.n_params())
            p[b.cam_slice(0)] = b.init_cam(cam, fd, td)
            for j, oid in enumerate(b.obj_ids):
                p[b.obj_slice(j)] = b.init_object(oid, p)
            try:
                r = b.solve(p, robust=False, max_nfev=200)
                r = b.solve(r.x, robust=True)
            except Exception as e:      # noqa: BLE001 — плохой старт, пропускаем
                if verbose:
                    print(fd, td, "fail", e)
                continue
            # Итоговая робастная стоимость, а не медиана: медиана прячет решение, где стены
            # и стол «промахнулись» (MISS), а дешёвые прямые сошлись.
            score = float(r.cost)
            if verbose:
                m, R, C = b.unpack_cam(r.x[b.cam_slice(0)])
                print(f"start f{fd} t{td}: cost {score:.1f} h={-C[2]:.2f}")
            if best is None or score < best[0]:
                best = (score, b, r.x)
    if best is None:
        raise RuntimeError(f"{cam.camera_id}: калибровка не сошлась")
    return best[1], best[2]


def to_camera(b: Bundle, p, i, extra=None) -> Camera:
    """Параметры bundle -> Camera с гомографией пола (совместима с остальным кодом)."""
    c = b.cams[i]
    m, R, C = b.unpack_cam(p[b.cam_slice(i)])
    t = -R @ C
    G = np.column_stack([R[:, 0], R[:, 1], t])       # пол (X, Y, 1) -> идеальные (не нормированные)
    H = np.linalg.inv(G)
    cam = Camera(c.camera_id, c.image_size, None, m)
    # Знак H: w > 0 для точек пола перед камерой.
    floor_uv = np.array([[c.image_size[0] / 2, c.image_size[1] * 0.75]])
    ideal, _ = cam.to_ideal(floor_uv)
    cam.H = normalize_H_sign(H, ideal)
    cam.raw["pose"] = {"rvec": cv2.Rodrigues(R)[0].ravel().tolist(), "center_m": C.tolist(),
                       "height_m": float(-C[2])}
    if extra:
        cam.raw.update(extra)
    return cam


def report(b: Bundle, p) -> dict:
    out = {}
    for i, c in enumerate(b.cams):
        parts = b.cam_residuals(i, p, detail=True)
        m, R, C = b.unpack_cam(p[b.cam_slice(i)])
        axis = R.T @ np.array([0, 0, 1.0])
        rep = {"height_m": round(float(-C[2]), 3),
               "center_m": [round(float(v), 2) for v in C[:2]],
               "tilt_deg": round(float(np.degrees(np.arccos(np.clip(axis[2], -1, 1)))), 1),
               "f": round(m.f, 1), "k": [round(v, 4) for v in m.k[:N_K]]}
        for k, v in parts.items():
            if k == "prior":
                continue
            vv = v.reshape(-1, 2) if k not in ("lines", "verticals") else v[:, None]
            e = np.linalg.norm(vv, axis=1)
            rep[k + "_px"] = {"n": int(len(e)), "median": round(float(np.median(e)), 2),
                              "max": round(float(e.max()), 2)}
        out[c.camera_id] = rep
    for j, oid in enumerate(b.obj_ids):
        q = p[b.obj_slice(j)]
        out[oid] = {"center_m": [round(float(q[0]), 2), round(float(q[1]), 2)],
                    "angle_deg": round(float(np.degrees(q[2])), 1)}
    for j, lid in enumerate(b.lm_ids):
        out["lm:" + lid] = [round(float(v), 2) for v in p[b.lm_slice(j)]]
    return out


def load_obs(path, m_per_px) -> CamObs:
    return CamObs.from_json(json.loads(Path(path).read_text(encoding="utf-8")), m_per_px)


def save_camera(b: Bundle, p, i, plan, path, status: str = "ok", note: str = "", extra_obs=None):
    """Камера из bundle -> calib/<cam>.json (H, дисторсия, поза, отчёт, зона обзора на плане)."""
    from .zones import raster_to_polygon, visible_floor
    cam = to_camera(b, p, i)
    rep = report(b, p)[b.cams[i].camera_id]
    vis = visible_floor(cam, plan)
    d = cam.to_dict()
    d.update({"calib_method": "bundle: вертикали, прямые, стены плана, предметы известного размера",
              "calib_status": status, "calib_note": note, "calib_report": rep,
              "fov_plan": raster_to_polygon(vis, 0.1), "fov_area_m2": round(float(vis.sum() * 0.01), 1)})
    if extra_obs:
        d["observations"] = extra_obs
    Path(path).write_text(json.dumps(d, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return cam, vis


# ------------------------------------------------------------------ уточнение по людям

def cam_params_from_camera(cam: Camera) -> np.ndarray:
    m = cam.distortion
    R = cv2.Rodrigues(np.asarray(cam.raw["pose"]["rvec"], float))[0]
    C = np.asarray(cam.raw["pose"]["center_m"], float)
    return np.r_[m.f, m.cx, m.cy, m.k[:N_K], cv2.Rodrigues(R)[0].ravel(), C]


def select_spread(pairs, n_max: int = 300, cell_m: float = 0.5, seed: int = 0):
    """Пары, равномерно разбросанные по полу: не больше нескольких на клетку cell_m."""
    if len(pairs) <= n_max:
        return pairs
    key = (np.floor(pairs["xa"] / cell_m).astype(int).astype(str) + "_" +
           np.floor(pairs["ya"] / cell_m).astype(int).astype(str))
    per = max(1, n_max // max(1, key.nunique()))
    sel = pairs.groupby(key, group_keys=False).apply(
        lambda g: g.sample(min(len(g), per), random_state=seed))
    return sel.head(n_max)


def refine_with_people(obs_dicts: dict, init_params: dict, pairs, m_per_px: float,
                       n_max: int = 300, person_w: float = 1.0, max_nfev: int = 150):
    """obs_dicts: {cam_id: наблюдения калибровки (json)}, init_params: {cam_id: вектор камеры},
    pairs: DataFrame из overlap.find_pairs (cam_a, cam_b, u_a, v_a, u_b, v_b, xa, ya).
    Каждая пара — общая точка пола, её положение неизвестно и подбирается вместе с камерами.
    -> (Bundle, параметры)."""
    pairs = select_spread(pairs, n_max).reset_index(drop=True)
    d = {c: dict(o) for c, o in obs_dicts.items()}
    for c in d:
        d[c]["landmarks"] = list(d[c].get("landmarks", []))
    for k, r in pairs.iterrows():
        lid = f"person_{k}"
        for side in ("a", "b"):
            c = r[f"cam_{side}"]
            if c in d:
                d[c]["landmarks"].append({"id": lid, "img": [float(r[f"u_{side}"]), float(r[f"v_{side}"])],
                                          "w": person_w})
    names = sorted(d)
    obs = [CamObs.from_json(d[c], m_per_px) for c in names]
    b = Bundle(obs, m_per_px)
    p = np.zeros(b.n_params())
    for i, c in enumerate(names):
        p[b.cam_slice(i)] = init_params[c]
    # Углы одного предмета на разных камерах могут быть пронумерованы с разных сторон
    # (поворот на 180°): согласуем по направлению, которое даёт каждая камера.
    for oid in b.obj_ids:
        ref = None
        for i, o in enumerate(obs):
            for k, (oo, img) in enumerate(o.objects):
                if oo != oid:
                    continue
                m, R, C = b.unpack_cam(p[b.cam_slice(i)])
                X, ok = backproject_floor(m, R, C, img, Z=-OBJECTS[oid]["height_m"])
                ang = np.arctan2(*(X[1] - X[0])[::-1])
                if ref is None:
                    ref = ang
                elif abs((ang - ref + np.pi) % (2 * np.pi) - np.pi) > np.pi / 2:
                    o.objects[k] = (oo, np.roll(img, 2, axis=0))
    for j, oid in enumerate(b.obj_ids):
        p[b.obj_slice(j)] = b.init_object(oid, p)
    # Общие точки: середина двух текущих проекций.
    for j, lid in enumerate(b.lm_ids):
        k = int(lid.split("_")[1]) if lid.startswith("person_") else None
        if k is not None:
            r = pairs.iloc[k]
            p[b.lm_slice(j)] = [(r["xa"] + r["xb"]) / 2, (r["ya"] + r["yb"]) / 2]
    print(f"  подбор: камер {len(names)}, общих точек {len(b.lm_ids)}, параметров {b.n_params()}", flush=True)
    r = b.solve(p, max_nfev=max_nfev, verbose=2)
    return b, r.x
