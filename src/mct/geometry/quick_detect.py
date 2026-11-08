"""Быстрые детекции людей для калибровки — пока У1 не выдал полный S1.

    pip install ultralytics
    python -m mct.geometry.quick_detect --videos data/videos --out outputs/calib_run --start 60 --seconds 120

Пишет outputs/calib_run/detections.parquet в формате контракта У1:
    det_id, camera_id, frame_idx, t, x1, y1, x2, y2, conf, mask_rle, mask_area
Пиксели — в исходном разрешении видео, маски — COCO RLE JSON-строкой, t — секунды от начала.
Кадры берутся по времени (сетка step_s), а не по номеру: у камеры 4 другой fps.
Это не замена S1: трекинга нет, нужен только для калибровки по людям и проверки фильтра.

Имя камеры берётся из имени файла: camera_new_1.mkv -> cam1 (как в calib/).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .rle import encode


def cam_name(path: Path) -> str:
    m = re.search(r"(\d+)\D*$", path.stem)
    return f"cam{m.group(1)}" if m else path.stem


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default="data/videos")
    ap.add_argument("--out", default="outputs/calib_run")
    ap.add_argument("--start", type=float, default=60.0, help="с какой секунды")
    ap.add_argument("--seconds", type=float, default=120.0, help="сколько секунд")
    ap.add_argument("--step", type=float, default=0.2, help="шаг по времени, с (0.2 = 5 кадров/с)")
    ap.add_argument("--model", default="yolo11s-seg.pt", help="веса Ultralytics (скачаются сами)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default=None, help="mps (Mac), cuda, cpu; по умолчанию — само")
    ap.add_argument("--cams", default=None, help="например cam1,cam2,cam3,cam5")
    a = ap.parse_args(argv)

    from ultralytics import YOLO
    model = YOLO(a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vids = sorted(p for p in Path(a.videos).iterdir() if p.suffix.lower() in (".mkv", ".mp4", ".avi", ".mov"))
    want = set(a.cams.split(",")) if a.cams else None
    rows, det_id = [], 0
    for v in vids:
        cam = cam_name(v)
        if want and cam not in want:
            continue
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS)
        times = np.arange(a.start, a.start + a.seconds, a.step)
        print(f"{cam}: {v.name}, {fps:.0f} fps, {len(times)} кадров")
        for k, t in enumerate(times):
            fi = int(round(t * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                break
            H, W = frame.shape[:2]
            res = model.predict(frame, classes=[0], conf=a.conf, imgsz=a.imgsz, device=a.device,
                                retina_masks=False, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            polys = res.masks.xy if res.masks is not None else [None] * len(boxes)
            for b, c, poly in zip(boxes, confs, polys):
                rle, area = None, 0
                if poly is not None and len(poly) >= 3:
                    m = np.zeros((H, W), np.uint8)
                    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
                    area = int(m.sum())
                    if area > 0:
                        rle = json.dumps(encode(m.astype(bool)))
                rows.append((det_id, cam, fi, round(fi / fps, 4), *[float(x) for x in b], float(c), rle, area))
                det_id += 1
            if k % 50 == 0:
                print(f"  {cam} t={t:.1f} с, детекций всего {det_id}")
        cap.release()
    df = pd.DataFrame(rows, columns=["det_id", "camera_id", "frame_idx", "t", "x1", "y1", "x2", "y2",
                                     "conf", "mask_rle", "mask_area"])
    df.to_parquet(out / "detections.parquet", index=False)
    print(df.groupby("camera_id").size().to_string())
    print(f"-> {out / 'detections.parquet'}")


if __name__ == "__main__":
    main()
