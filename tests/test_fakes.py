import json
from pathlib import Path

import numpy as np

from mct.geometry.fakes import simulate
from mct.geometry.plan import Plan

PLAN = Plan.load(Path(__file__).with_name("plan_fixture.json"))


def test_fakes_contract_and_consistency():
    r = simulate(PLAN, minutes=2, seed=3)
    tr, trk, td, gt = r["trajectories"], r["tracklets"], r["tracklet_dets"], r["gt"]
    assert list(tr.columns) == ["det_id", "tracklet_id", "t", "x", "y", "sigma", "zone", "uncertain",
                                "vx", "vy", "cams_cover"]
    assert list(trk.columns) == ["tracklet_id", "camera_id", "t_start", "t_end", "n_dets", "local_ids"]
    assert tr["det_id"].is_unique and set(td["tracklet_id"]) == set(trk["tracklet_id"])
    n = tr.groupby("tracklet_id").size()
    assert (n.loc[trk["tracklet_id"]].to_numpy() == trk["n_dets"].to_numpy()).all()
    # Точка трека лежит в зоне обзора своей камеры (с учётом шума — почти всегда).
    cam = tr.merge(trk[["tracklet_id", "camera_id"]], on="tracklet_id")
    inside = [c in json.loads(cc) for c, cc in zip(cam["camera_id"], cam["cams_cover"])]
    assert np.mean(inside) > 0.9
    # Строгий режим даёт не меньше треков, чем transit.
    assert gt["gt_strict"].nunique() >= gt["gt_transit"].nunique() > 0
    assert set(gt["tracklet_id"]) == set(trk["tracklet_id"])


def test_fakes_are_reproducible():
    a = simulate(PLAN, minutes=1, seed=5)["trajectories"]
    b = simulate(PLAN, minutes=1, seed=5)["trajectories"]
    assert a.equals(b)


def test_graph_has_corridor_gaps():
    g = simulate(PLAN, minutes=0.5, seed=0)["graph"]
    ids = {x["id"] for x in g["gaps"]}
    assert {"gap_cam3_cam4", "gap_cam4_cam5"} <= ids
    for x in g["gaps"]:
        assert 0 < x["t_min_s"] < x["t_max_s"]
    assert any(set(o["cams"]) == {"cam1", "cam2"} for o in g["overlaps"])
