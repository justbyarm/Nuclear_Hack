"""Кадры для калибровки: медианный кадр без людей и одиночный кадр по времени.

    python -m mct.geometry.frames median --video data/cam1.mp4 --out data/frames/cam1_median.png
    python -m mct.geometry.frames grab   --video data/cam1.mp4 --t 60 --out data/frames/cam1_t60.png

Медиана по ~60 кадрам за весь ролик убирает людей: пол виден целиком,
клики точнее. Кадр сохраняется в ИСХОДНОМ разрешении видео — это важно,
пиксели калибровки должны совпадать с пикселями детекций.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _open(video):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"не открывается видео: {video}")
    return cap


def video_info(video) -> dict:
    cap = _open(video)
    info = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    cap.release()
    info["duration_s"] = info["n_frames"] / info["fps"] if info["fps"] else None
    return info


def grab(video, t: float) -> np.ndarray:
    cap = _open(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"не удалось прочитать кадр на {t} с")
    return frame


def median_frame(video, n: int = 60, t0: float = 0.0, t1: float | None = None) -> np.ndarray:
    info = video_info(video)
    t1 = t1 if t1 is not None else (info["duration_s"] or 60) - 0.5
    cap = _open(video)
    frames = []
    for t in np.linspace(t0, t1, n):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    if len(frames) < 5:
        raise RuntimeError("слишком мало кадров прочиталось для медианы")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def median_of_images(images) -> np.ndarray:
    return np.median(np.stack(list(images)), axis=0).astype(np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("median"); m.add_argument("--video", required=True); m.add_argument("--out", required=True)
    m.add_argument("--n", type=int, default=60)
    g = sub.add_parser("grab"); g.add_argument("--video", required=True); g.add_argument("--out", required=True)
    g.add_argument("--t", type=float, default=0.0)
    i = sub.add_parser("info"); i.add_argument("--video", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "info":
        print(video_info(a.video)); return
    img = median_frame(a.video, a.n) if a.cmd == "median" else grab(a.video, a.t)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, img)
    print(f"{a.out}: {img.shape[1]}x{img.shape[0]}")


if __name__ == "__main__":
    main()
