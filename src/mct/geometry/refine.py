"""Уточнение калибровки по людям — одна команда после детекций У1 и S1b.

    python -m mct.geometry.refine --run outputs/run --calib calib            # отчёт, ничего не меняет
    python -m mct.geometry.refine --run outputs/run --calib calib --apply    # перезаписать calib/*.json

Шаги:
  1. Пары «один человек на двух камерах» по текущей калибровке (+ поиск сдвига времени).
  2. Совместный подбор всех камер: прежние наблюдения (прямые, вертикали, стены, стол) + пары людей.
  3. Пары пересчитываются по новой калибровке — сравнение «до / после», D_max для У3.
  4. С --apply: старые json -> calib/backup_<время>/, новые json, новые fov_plan и graph.json.
Нужно: outputs/run/detections.parquet, outputs/run/floor_check.parquet (S1b).
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd

from .bundle import cam_params_from_camera, refine_with_people, report, save_camera
from .floor_check import load_cameras, run as floor_run
from .graph import build_graph, save_graph
from .overlap import analyze
from .plan import Plan


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--calib", default="calib")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--n-max", type=int, default=250, help="сколько пар людей брать в подбор")
    a = ap.parse_args(argv)
    run, cdir = Path(a.run), Path(a.calib)
    dets = pd.read_parquet(run / "detections.parquet")
    plan = Plan.load(cdir / "plan.json")
    cams = load_cameras(cdir)
    fc = floor_run(dets, cams, plan)
    before, pairs = analyze(dets, fc, cams, plan)
    print("ДО:", json.dumps(before, ensure_ascii=False, indent=1))
    if pairs.empty:
        print("пар нет: камеры не перекрываются по текущей калибровке или мало детекций")
        return
    obs = {c: cams[c].raw["observations"] for c in cams if cams[c].raw.get("observations")}
    missing = sorted(set(cams) - set(obs))
    if missing:
        print(f"нет сохранённых наблюдений калибровки у {missing}: эти камеры не уточняются")
    init = {c: cam_params_from_camera(cams[c]) for c in obs}
    pairs = pairs[pairs["cam_a"].isin(obs) & pairs["cam_b"].isin(obs)]
    t0 = time.time()
    print(f"[1/3] подбор калибровки по {min(len(pairs), a.n_max)} парам людей…", flush=True)
    b, p = refine_with_people(obs, init, pairs, plan.m_per_px, n_max=a.n_max)
    print(f"      готово за {time.time() - t0:.0f} с", flush=True)
    rep = report(b, p)
    out = cdir if a.apply else run / "calib_refined"
    if a.apply:
        bk = cdir / f"backup_{time.strftime('%Y%m%d_%H%M%S')}"
        bk.mkdir()
        for f in cdir.glob("*.json"):
            shutil.copy(f, bk / f.name)
        print(f"старая калибровка -> {bk}")
    else:
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(cdir / "plan.json", out / "plan.json")
    fovs = {}
    print("[2/3] зоны обзора и граф…", flush=True)
    for i, o in enumerate(b.cams):
        c = o.camera_id
        save_camera(b, p, i, plan, out / f"{c}.json", status="refined_people",
                    note=f"уточнено по {len(pairs)} парам людей", extra_obs=obs[c])
        fovs[c] = json.loads((out / f"{c}.json").read_text(encoding="utf-8"))["fov_plan"]
    save_graph(build_graph(plan, fovs), out / "graph.json")
    print("[3/3] проверка: пары по новой калибровке…", flush=True)
    cams2 = load_cameras(out)
    after, _ = analyze(dets, floor_run(dets, cams2, plan), cams2, plan)
    print("ПОСЛЕ:", json.dumps(after, ensure_ascii=False, indent=1))
    print("камеры:", json.dumps({k: v for k, v in rep.items() if k.startswith("cam")}, ensure_ascii=False, default=float))
    (run / "refine_report.json").write_text(json.dumps({"before": before, "after": after, "cameras": rep},
                                                       ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"-> {out}" + ("" if a.apply else "  (проверь и запусти с --apply)"))


if __name__ == "__main__":
    main()
