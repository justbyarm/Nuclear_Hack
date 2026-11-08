"""Инструмент калибровки: клики по кадру и плану в окнах OpenCV.

Камера (кадр и план в двух окнах):
    python -m mct.geometry.tool cam --camera cam1 --frame data/frames/cam1_median.png \
        --plan-image data/plan_clean.png --plan calib/plan.json

План (масштаб, пол, выходы, препятствия, опорные точки коридора):
    python -m mct.geometry.tool plan --plan-image data/plan_clean.png --plan calib/plan.json

Управление (одинаково в обоих окнах):
    колесо / i, o   зум к курсору        правая кнопка + тянуть   сдвиг
    z               сбросить вид         Enter / n               закончить линию или полигон
    Backspace       отменить последнее   s                       сохранить
    h               подсказка в консоль  q / Esc                 выйти (сохраняет)

Режимы камеры:  l — прямые (швы плитки, края стен у пола, косяки), p — пары кадр↔план,
                f — видимый пол на кадре, d — столы и всё, что закрывает ноги, e — выходы на кадре,
                c — посчитать калибровку и показать сетку 1×1 м.
Режимы плана:   k — отрезок известной длины (масштаб), f — пол, x — выход (1–5 тип), o — препятствие,
                g — опорные точки середины коридора.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import polygons as P
from . import viz
from .calibrate import calibrate_camera
from .camera import Camera
from .plan import Plan, scale_from_segments

COL = {  # BGR
    "line": (255, 160, 0), "line_cur": (255, 255, 0), "pair": (0, 220, 0), "pair_pending": (0, 0, 255),
    "floor": (0, 200, 0), "desk": (0, 140, 255), "exit": (255, 0, 255), "occ": (0, 0, 220),
    "nav": (255, 255, 0), "scale": (0, 0, 255), "grid": (0, 255, 255), "pred": (0, 0, 255),
}
EXIT_TYPES = {ord("1"): "door", ord("2"): "toilet", ord("3"): "entrance", ord("4"): "stairs",
              ord("5"): "corridor_out"}


# ===================================================================== вид с зумом

@dataclass
class View:
    """Окно-лупа над большой картинкой: зум к курсору, сдвиг, перевод координат."""
    img_w: int
    img_h: int
    win_w: int = 1280
    win_h: int = 900
    zoom: float = 1.0          # 1 = картинка целиком вписана в окно
    cx: float | None = None    # центр вида в пикселях картинки
    cy: float | None = None

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.zoom, self.cx, self.cy = 1.0, self.img_w / 2, self.img_h / 2

    @property
    def scale(self):  # пиксели окна на пиксель картинки
        return self.zoom * min(self.win_w / self.img_w, self.win_h / self.img_h)

    def to_img(self, x, y):
        s = self.scale
        return self.cx + (x - self.win_w / 2) / s, self.cy + (y - self.win_h / 2) / s

    def to_win(self, u, v):
        s = self.scale
        return (u - self.cx) * s + self.win_w / 2, (v - self.cy) * s + self.win_h / 2

    def zoom_at(self, x, y, factor):
        u, v = self.to_img(x, y)
        self.zoom = float(np.clip(self.zoom * factor, 1.0, 40.0))
        s = self.scale
        self.cx, self.cy = u - (x - self.win_w / 2) / s, v - (y - self.win_h / 2) / s

    def pan(self, dx, dy):
        self.cx -= dx / self.scale
        self.cy -= dy / self.scale

    def render(self, img):
        s = self.scale
        M = np.array([[s, 0, self.win_w / 2 - self.cx * s], [0, s, self.win_h / 2 - self.cy * s]])
        interp = cv2.INTER_NEAREST if s > 3 else cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
        return cv2.warpAffine(img, M, (self.win_w, self.win_h), flags=interp,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))


class Window:
    def __init__(self, name, img, on_click, win_size=(1280, 900)):
        self.name, self.img, self.on_click = name, img, on_click
        h, w = img.shape[:2]
        self.view = View(w, h, *win_size)
        self._drag = None
        self.status = ""
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(name, self._mouse)

    def _mouse(self, ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            u, v = self.view.to_img(x, y)
            self.on_click(float(u), float(v))
        elif ev == cv2.EVENT_RBUTTONDOWN:
            self._drag = (x, y)
        elif ev == cv2.EVENT_MOUSEMOVE and self._drag and flags & cv2.EVENT_FLAG_RBUTTON:
            self.view.pan(x - self._drag[0], y - self._drag[1]); self._drag = (x, y)
        elif ev == cv2.EVENT_RBUTTONUP:
            self._drag = None
        elif ev in (cv2.EVENT_MOUSEWHEEL, getattr(cv2, "EVENT_MOUSEHWHEEL", -99)):
            self.view.zoom_at(x, y, 1.25 if cv2.getMouseWheelDelta(flags) > 0 else 0.8)
        self.last_mouse = (x, y)

    def show(self, overlay):
        im = self.view.render(overlay)
        cv2.rectangle(im, (0, 0), (self.view.win_w, 28), (0, 0, 0), -1)
        cv2.putText(im, self.status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(self.name, im)


# ===================================================================== калибровка камеры

class CamSession:
    """Вся логика кликов и расчёта для одной камеры. Без окон — тестируется напрямую."""

    MODES = {"l": "прямые", "p": "пары", "f": "пол на кадре", "d": "столы", "e": "выходы на кадре"}

    def __init__(self, camera_id, frame, plan_img, plan: Plan | None, calib_path):
        self.camera_id, self.frame, self.plan_img, self.plan = camera_id, frame, plan_img, plan
        self.calib_path = Path(calib_path)
        self.mode = "l"
        self.lines, self.cur = [], []          # cur — текущая линия/полигон
        self.pairs = []                        # {"img": [u, v], "plan_px": [x, y]|None, "label"}
        self.floor_img, self.desks, self.exits = [], [], []
        self.result = None
        self.message = ""
        self._dirty = True
        if self.calib_path.exists():
            self.load()

    # --- клики
    def click_frame(self, u, v):
        if self.mode == "p":
            if self.pairs and self.pairs[-1]["plan_px"] is None:
                self.pairs[-1]["img"] = [u, v]           # перекликнули точку кадра
            else:
                self.pairs.append({"img": [u, v], "plan_px": None, "label": str(len(self.pairs) + 1)})
            self.message = f"пара {len(self.pairs)}: теперь кликни ту же точку на плане"
        else:
            self.cur.append([u, v])
        self._dirty = True

    def click_plan(self, x, y):
        if self.mode != "p":
            self.message = "на плане кликаем только в режиме пар (p)"; return
        if not self.pairs or self.pairs[-1]["plan_px"] is not None:
            self.message = "сначала точку на КАДРЕ, потом на плане"; return
        self.pairs[-1]["plan_px"] = [x, y]
        self.message = f"пара {len(self.pairs)} готова"
        self._dirty = True

    def finish(self):
        """Закрыть текущую линию или полигон."""
        if self.mode == "l" and len(self.cur) >= 3:
            self.lines.append(self.cur)
        elif self.mode == "f" and len(self.cur) >= 3:
            self.floor_img = self.cur
        elif self.mode == "d" and len(self.cur) >= 3:
            self.desks.append(self.cur)
        elif self.mode == "e" and len(self.cur) >= 3:
            self.exits.append(self.cur)
        elif self.cur:
            self.message = "мало точек: у линии и полигона нужно хотя бы 3"
            return
        self.cur = []
        self._dirty = True

    def undo(self):
        if self.cur:
            self.cur.pop()
        elif self.mode == "p" and self.pairs:
            if self.pairs[-1]["plan_px"] is not None:
                self.pairs[-1]["plan_px"] = None
            else:
                self.pairs.pop()
        elif self.mode == "l" and self.lines:
            self.cur = self.lines.pop()
        elif self.mode == "d" and self.desks:
            self.cur = self.desks.pop()
        elif self.mode == "e" and self.exits:
            self.cur = self.exits.pop()
        elif self.mode == "f" and self.floor_img:
            self.cur, self.floor_img = self.floor_img, []
        self._dirty = True

    def set_mode(self, m):
        if self.cur:
            self.finish()
        self.mode, self._dirty = m, True

    # --- расчёт
    def complete_pairs(self):
        return [p for p in self.pairs if p["img"] is not None and p["plan_px"] is not None]

    def calibrate(self):
        if self.plan is None:
            self.message = "нет plan.json с масштабом — сначала размечаем план"; return None
        pairs = self.complete_pairs()
        h, w = self.frame.shape[:2]
        try:
            res = calibrate_camera(self.camera_id, (w, h), [p["img"] for p in pairs],
                                   self.plan.px_to_m([p["plan_px"] for p in pairs]), self.lines)
        except (ValueError, RuntimeError) as e:
            self.message = f"не посчиталось: {e}"; return None
        self.result = res
        s = res.summary()
        self.message = (f"h={s['height_m']} м, наклон {s['tilt_deg']}°, пары {s['n_inliers']}/{s['n_pairs']}, "
                        f"ош. {s['pair_mean_m']}/{s['pair_max_m']} м, LOO {s['loo_mean_m']}/{s['loo_max_m']} м, "
                        f"прямые {s['line_rms_px']} px")
        for n in s["notes"]:
            self.message += f" | {n}"
        self._dirty = True
        return res

    def fov_plan(self):
        """Видимый пол кадра на плане. Стороны уплотняем: под fisheye они на плане кривые."""
        dense = P.densify(self.floor_img, step=15.0)
        fov, ok = self.result.camera.project(dense)
        return fov[ok]

    # --- json
    def to_dict(self):
        h, w = self.frame.shape[:2]
        d = {"camera_id": self.camera_id, "image_size": [w, h], "H": None, "distortion": None}
        if self.result is not None:
            d.update(self.result.camera.to_dict())
            d["calib_summary"] = self.result.summary()
            d["pair_errors_m"] = [round(float(e), 4) for e in self.result.pair_err_m]
            d["pair_inliers"] = [bool(b) for b in self.result.inliers]
            if self.floor_img and self.plan is not None:
                d["fov_plan"] = self.fov_plan().round(3).tolist()
        d["clicks"] = {"lines": self.lines, "pairs": self.pairs, "floor_img": self.floor_img,
                       "desk_zones_img": self.desks, "exit_zones_img": self.exits}
        d["floor_img"], d["desk_zones_img"], d["exit_zones_img"] = self.floor_img, self.desks, self.exits
        return d

    def save(self):
        self.calib_path.parent.mkdir(parents=True, exist_ok=True)
        self.calib_path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        self.message = f"сохранено: {self.calib_path}"
        return self.calib_path

    def load(self):
        d = json.loads(self.calib_path.read_text(encoding="utf-8"))
        c = d.get("clicks", {})
        self.lines, self.pairs = c.get("lines", []), c.get("pairs", [])
        self.floor_img = c.get("floor_img", [])
        self.desks, self.exits = c.get("desk_zones_img", []), c.get("exit_zones_img", [])
        self.message = f"загружено из {self.calib_path}: {len(self.lines)} прямых, {len(self.pairs)} пар"

    # --- картинки
    def status(self):
        n = len(self.complete_pairs())
        return f"[{self.camera_id}] режим: {self.MODES[self.mode]} | прямых {len(self.lines)} | пар {n} | {self.message}"

    def render_frame(self):
        im = self.frame.copy()
        if self.result is not None:
            cam = self.result.camera
            XY = self.plan.px_to_m(self._plan_extent_px()) if self.plan else None
            if XY is not None:
                im = viz.draw_grid(im, cam, (*XY[0], *XY[1]), 1.0, COL["grid"], 2,
                                   floor_polys=self.plan.floor or None)
        for l in self.lines:
            cv2.polylines(im, [np.round(np.array(l)).astype(np.int32)], False, COL["line"], 2, cv2.LINE_AA)
        viz.draw_poly(im, self.floor_img, COL["floor"], fill_alpha=0.12) if self.floor_img else None
        for p in self.desks:
            viz.draw_poly(im, p, COL["desk"], fill_alpha=0.2)
        for p in self.exits:
            viz.draw_poly(im, p, COL["exit"], fill_alpha=0.2)
        cur_col = {"l": COL["line_cur"], "f": COL["floor"], "d": COL["desk"], "e": COL["exit"]}.get(self.mode)
        if self.cur and cur_col:
            viz.draw_poly(im, self.cur, cur_col, closed=self.mode != "l")
        for i, p in enumerate(self.pairs):
            col = COL["pair"] if p["plan_px"] is not None else COL["pair_pending"]
            viz.draw_points(im, [p["img"]], col, 7, [p["label"]])
        # Куда модель кладёт точку плана — красный крестик рядом с кликом = видна ошибка.
        if self.result is not None:
            done = self.complete_pairs()
            uv, ok = self.result.camera.unproject(self.plan.px_to_m([p["plan_px"] for p in done]))
            for (u, v), good in zip(uv, ok):
                if good:
                    cv2.drawMarker(im, (int(u), int(v)), COL["pred"], cv2.MARKER_CROSS, 14, 2)
        return im

    def render_plan(self):
        im = self.plan_img.copy()
        if self.plan is not None:
            for poly in self.plan.raw.get("px", {}).get("floor", []):
                viz.draw_poly(im, poly, COL["floor"], thickness=1)
        if self.result is not None and self.floor_img:
            fov = self.fov_plan()
            if len(fov) >= 3:
                viz.draw_poly(im, self.plan.m_to_px(fov), COL["grid"], fill_alpha=0.15)
        for p in self.pairs:
            if p["plan_px"] is not None:
                viz.draw_points(im, [p["plan_px"]], COL["pair"], 5, [p["label"]])
        if self.result is not None and self.plan is not None:
            c = self.plan.m_to_px(np.array(self.result.camera.raw["pose"]["center_m"][:2]))
            cv2.drawMarker(im, (int(c[0]), int(c[1])), (0, 0, 255), cv2.MARKER_TRIANGLE_UP, 18, 3)
        return im

    def _plan_extent_px(self):
        h, w = self.plan_img.shape[:2]
        return np.array([[0, 0], [w, h]], float)


# ===================================================================== разметка плана

class PlanSession:
    MODES = {"k": "масштаб", "f": "пол", "x": "выходы", "o": "препятствия", "g": "опорные точки коридора"}

    def __init__(self, plan_img, plan_path, plan_image_path="", seg_m=1.2):
        self.img, self.path, self.seg_m = plan_img, Path(plan_path), seg_m
        self.plan_image_path = str(plan_image_path)
        self.mode, self.cur, self.message = "k", [], ""
        self.segments, self.floor, self.exits, self.occ, self.nav = [], [], [], [], []
        if self.path.exists():
            self.load()

    def click(self, x, y):
        if self.mode == "g":
            self.nav.append([x, y])
        else:
            self.cur.append([x, y])
            if self.mode == "k" and len(self.cur) == 2:
                self.segments.append({"px": self.cur, "length_m": self.seg_m}); self.cur = []
                self.message = self._scale_msg()

    def finish(self):
        if len(self.cur) >= 3:
            if self.mode == "f":
                self.floor.append(self.cur)
            elif self.mode == "x":
                self.exits.append({"id": f"exit_{len(self.exits) + 1}", "type": "door", "polygon": self.cur})
                self.message = f"{self.exits[-1]['id']}: тип door (1 door, 2 toilet, 3 entrance, 4 stairs, 5 corridor_out)"
            elif self.mode == "o":
                self.occ.append({"id": f"occ_{len(self.occ) + 1}", "polygon": self.cur})
        self.cur = []

    def set_exit_type(self, key):
        if self.exits and key in EXIT_TYPES:
            self.exits[-1]["type"] = EXIT_TYPES[key]
            self.message = f"{self.exits[-1]['id']}: тип {EXIT_TYPES[key]}"

    def undo(self):
        if self.cur:
            self.cur.pop(); return
        lst = {"k": self.segments, "f": self.floor, "x": self.exits, "o": self.occ, "g": self.nav}[self.mode]
        if lst:
            lst.pop()

    def set_mode(self, m):
        self.finish(); self.mode = m

    def mpp(self) -> float:
        return scale_from_segments([g["px"] for g in self.segments],
                                   [g["length_m"] for g in self.segments])["m_per_px"]

    def _scale_msg(self):
        s = scale_from_segments([g["px"] for g in self.segments], [g["length_m"] for g in self.segments])
        return f"масштаб {s['m_per_px']:.5f} м/px по {s['n']} отрезкам, разброс {s['spread_pct']:.1f}%"

    def build(self) -> Plan:
        if not self.segments:
            raise ValueError("нет отрезков масштаба (режим k)")
        s = scale_from_segments([g["px"] for g in self.segments], [g["length_m"] for g in self.segments])
        mpp = s["m_per_px"]
        to_m = lambda pts: (np.asarray(pts, float) * mpp).round(4).tolist()
        h, w = self.img.shape[:2]
        raw = {"scale_method": f"медиана по {s['n']} отрезкам известной длины, разброс {s['spread_pct']:.1f}%",
               "scale_segments": self.segments,
               "px": {"floor": self.floor, "exits": self.exits, "occluders": self.occ, "nav_nodes": self.nav}}
        return Plan(mpp, self.plan_image_path, (round(w * mpp, 3), round(h * mpp, 3)),
                    [to_m(p) for p in self.floor],
                    [{**e, "polygon": to_m(e["polygon"])} for e in self.exits],
                    [{**o, "polygon": to_m(o["polygon"])} for o in self.occ],
                    to_m(self.nav) if self.nav else [], raw)

    def save(self):
        plan = self.build()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        plan.save(self.path)
        self.message = f"сохранено: {self.path}"
        return plan

    def load(self):
        d = json.loads(self.path.read_text(encoding="utf-8"))
        px = d.get("px", {})
        self.segments = d.get("scale_segments", [])
        self.floor, self.exits = px.get("floor", []), px.get("exits", [])
        self.occ, self.nav = px.get("occluders", []), px.get("nav_nodes", [])
        self.message = f"загружено из {self.path}"

    def status(self):
        return (f"[план] режим: {self.MODES[self.mode]} | отрезков {len(self.segments)} | пол {len(self.floor)} | "
                f"выходов {len(self.exits)} | препятствий {len(self.occ)} | опорных {len(self.nav)} | {self.message}")

    def render(self):
        im = self.img.copy()
        for g in self.segments:
            a, b = np.round(g["px"]).astype(int)
            cv2.line(im, tuple(a), tuple(b), COL["scale"], 2, cv2.LINE_AA)
        for p in self.floor:
            viz.draw_poly(im, p, COL["floor"], fill_alpha=0.12)
        for e in self.exits:
            viz.draw_poly(im, e["polygon"], COL["exit"], fill_alpha=0.25)
            c = np.mean(e["polygon"], 0)
            cv2.putText(im, e["type"], (int(c[0]), int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL["exit"], 1, cv2.LINE_AA)
        for o in self.occ:
            viz.draw_poly(im, o["polygon"], COL["occ"], fill_alpha=0.25)
        if self.nav:
            # Опорные точки — граф, а не одна ломаная: соединяем только близких соседей,
            # иначе на развилках рисуются ложные линии сквозь стены.
            pts = np.asarray(self.nav, float)
            link = 3.0 / self.mpp() if self.segments else 60.0
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    if np.linalg.norm(pts[i] - pts[j]) <= link:
                        cv2.line(im, tuple(np.round(pts[i]).astype(int)), tuple(np.round(pts[j]).astype(int)),
                                 COL["nav"], 1, cv2.LINE_AA)
            viz.draw_points(im, pts, COL["nav"], 3, thickness=-1)
        if self.cur:
            viz.draw_poly(im, self.cur, (0, 0, 255), closed=False)
        return im


# ===================================================================== окна

def _load_img(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"не читается картинка: {path}")
    return img


def run_cam(a):
    frame, plan_img = _load_img(a.frame), _load_img(a.plan_image)
    plan = Plan.load(a.plan) if Path(a.plan).exists() else None
    s = CamSession(a.camera, frame, plan_img, plan, a.out or f"calib/{a.camera}.json")
    wf = Window(f"kadr {a.camera}", frame, s.click_frame)
    wp = Window("plan", plan_img, s.click_plan, (900, 900))
    print(__doc__)
    while True:
        wf.status = wp.status = s.status()
        wf.show(s.render_frame()); wp.show(s.render_plan())
        k = cv2.waitKey(30) & 0xFF
        if k == 255:
            continue
        if k in (ord("q"), 27):
            s.save(); break
        ch = chr(k) if k < 128 else ""
        if ch in CamSession.MODES:
            s.set_mode(ch)
        elif k in (13, 10) or ch == "n":
            s.finish()
        elif k == 8 or k == 127:
            s.undo()
        elif ch == "c":
            s.calibrate(); print(s.message)
        elif ch == "s":
            s.save(); print(s.message)
        elif ch in "io":
            for w in (wf, wp):
                x, y = getattr(w, "last_mouse", (w.view.win_w / 2, w.view.win_h / 2))
                w.view.zoom_at(x, y, 1.25 if ch == "i" else 0.8)
        elif ch == "z":
            wf.view.reset(); wp.view.reset()
        elif ch == "h":
            print(__doc__)
    cv2.destroyAllWindows()


def run_plan(a):
    img = _load_img(a.plan_image)
    s = PlanSession(img, a.plan, a.plan_image, a.seg_m)
    w = Window("plan", img, s.click, (1300, 950))
    print(__doc__)
    while True:
        w.status = s.status()
        w.show(s.render())
        k = cv2.waitKey(30) & 0xFF
        if k == 255:
            continue
        if k in (ord("q"), 27):
            try:
                s.save()
            except ValueError as e:
                print(f"не сохранено: {e}")
            break
        ch = chr(k) if k < 128 else ""
        if ch in PlanSession.MODES:
            s.set_mode(ch)
        elif k in EXIT_TYPES:
            s.set_exit_type(k)
        elif k in (13, 10) or ch == "n":
            s.finish()
        elif k in (8, 127):
            s.undo()
        elif ch == "s":
            try:
                s.save()
            except ValueError as e:
                s.message = str(e)
            print(s.message)
        elif ch in "io":
            x, y = getattr(w, "last_mouse", (w.view.win_w / 2, w.view.win_h / 2))
            w.view.zoom_at(x, y, 1.25 if ch == "i" else 0.8)
        elif ch == "z":
            w.view.reset()
        elif ch == "h":
            print(__doc__)
    cv2.destroyAllWindows()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cam")
    c.add_argument("--camera", required=True)
    c.add_argument("--frame", required=True, help="кадр в исходном разрешении видео")
    c.add_argument("--plan-image", required=True)
    c.add_argument("--plan", default="calib/plan.json")
    c.add_argument("--out", default=None, help="по умолчанию calib/<camera>.json")
    p = sub.add_parser("plan")
    p.add_argument("--plan-image", required=True)
    p.add_argument("--plan", default="calib/plan.json")
    p.add_argument("--seg-m", type=float, default=1.2, help="длина отрезка масштаба, м")
    a = ap.parse_args(argv)
    (run_cam if a.cmd == "cam" else run_plan)(a)


if __name__ == "__main__":
    main()
