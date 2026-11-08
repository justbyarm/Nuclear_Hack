"""graph.json: перекрытия камер, слепые участки между ними, выходы.

Слепой участок = связная часть пола, которую не видит ни одна камера и которая граничит
с зонами обзора хотя бы двух камер. Тупики (граничат с одной камерой) — тоже записываем,
у них b = null: человек туда может уйти, но вернётся через ту же камеру.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .zones import ZoneMap


def _segment_from_cells(cells_ij, res):
    """Отрезок вдоль клеток (порог между зонами): крайние точки по главной оси, метры."""
    P = cells_ij[:, ::-1].astype(float) * res
    if len(P) == 1:
        return [P[0].round(2).tolist(), P[0].round(2).tolist()]
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    s = (P - c) @ vt[0]
    return [(c + s.min() * vt[0]).round(2).tolist(), (c + s.max() * vt[0]).round(2).tolist()]


def build_graph(plan, fovs: dict, res: float = 0.1, v_max: float = 2.5, v_slow: float = 0.5,
                time_margin_s: float = 2.0, min_overlap_m2: float = 1.0, min_gap_m2: float = 0.5) -> dict:
    zm = ZoneMap(plan, fovs, res)
    cell = res * res
    out = {"res_m": res, "overlaps": [], "gaps": [], "exits": []}
    ids = zm.cam_ids
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            m = zm.cover[a] & zm.cover[b]
            area = float(m.sum() * cell)
            if area >= min_overlap_m2:
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                big = max(cs, key=cv2.contourArea).reshape(-1, 2) * res
                out["overlaps"].append({"cams": [ids[a], ids[b]], "area_m2": round(area, 2),
                                        "polygon": big.round(2).tolist()})
    covered = zm.cover.any(0) if len(ids) else np.zeros(zm.shape, bool)
    blind = zm.floor & ~covered
    n, lab = cv2.connectedComponents(blind.astype(np.uint8), connectivity=4)
    k3 = np.ones((3, 3), np.uint8)
    for g in range(1, n):
        comp = lab == g
        area = float(comp.sum() * cell)
        if area < min_gap_m2:
            continue
        ring = cv2.dilate(comp.astype(np.uint8), k3).astype(bool) & ~comp
        touch = [(ids[c], int((ring & zm.cover[c]).sum())) for c in range(len(ids))]
        touch = sorted([t for t in touch if t[1] > 0], key=lambda t: -t[1])
        if not touch:
            continue
        # Длина по главной оси компоненты (коридор) — оценка пути через слепой участок.
        ij = np.argwhere(comp)
        P = ij[:, ::-1] * res
        c = P.mean(0)
        _, _, vt = np.linalg.svd(P - c, full_matrices=False)
        s = (P - c) @ vt[0]
        length = float(s.max() - s.min() + res)
        doors = []
        grown = cv2.dilate(comp.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        for e_i, eid in enumerate(zm.exit_ids):
            if (grown & (zm.exit_idx == e_i)).any():
                doors.append(eid)
        cams = [t[0] for t in touch]
        gid = "gap_" + "_".join(cams[:2]) if len(cams) >= 2 else f"deadend_{cams[0]}_{g}"
        portals = {}
        for cid in cams[:2]:
            ci = ids.index(cid)
            portals[cid] = _segment_from_cells(np.argwhere(ring & zm.cover[ci]), res)
        out["gaps"].append({
            "id": gid, "a": cams[0], "b": cams[1] if len(cams) > 1 else None, "touch_cams": cams,
            "portal_a": portals[cams[0]], "portal_b": portals.get(cams[1]) if len(cams) > 1 else None,
            "area_m2": round(area, 2), "length_m": round(length, 2),
            "t_min_s": round(length / v_max, 2), "t_max_s": round(length / v_slow + time_margin_s, 2),
            "doors_inside": doors})
    for e_i, e in enumerate(plan.exits):
        m = zm.exit_idx == e_i
        cams = [ids[c] for c in range(len(ids)) if (zm.cover[c] & m).any()]
        out["exits"].append({"id": e["id"], "type": e.get("type", "door"), "cams": cams})
    return out


def save_graph(g: dict, path) -> None:
    Path(path).write_text(json.dumps(g, ensure_ascii=False, indent=1), encoding="utf-8")
