from pathlib import Path

import numpy as np
import pandas as pd

from mct.geometry.fakes import simulate
from mct.geometry.plan import Plan
from mct.geometry.track_path import Router, build, fuse, position_at

PLAN = Plan.load(Path(__file__).with_name("plan_fixture.json"))


def test_fuse_weights_by_sigma_and_picks_best_camera():
    t = np.arange(0, 1.01, 0.1)
    a = pd.DataFrame({"tracklet_id": 1, "t": t, "x": 1.0, "y": 0.0, "sigma": 0.1})
    b = pd.DataFrame({"tracklet_id": 2, "t": t, "x": 2.0, "y": 0.0, "sigma": 0.3})
    f = fuse(pd.concat([a, b]), {1: "cam1", 2: "cam2"})
    assert np.allclose(f["x"], (1 / 0.01 + 2 / 0.09) / (1 / 0.01 + 1 / 0.09))
    assert (f["camera"] == "cam1").all()


def test_router_goes_along_corridor_not_through_walls():
    r = Router(PLAN)
    m = PLAN.m_per_px
    a, b = np.array([437, 250]) * m, np.array([300, 517]) * m       # холл -> проход к 339
    pts, routed = r.route(a, b)
    assert routed and len(pts) > 3
    seg = np.vstack([p0 + (p1 - p0) * np.linspace(0, 1, 20)[:, None] for p0, p1 in zip(pts[:-1], pts[1:])])
    assert PLAN.on_floor(seg, 0.35).all()


def test_build_track_from_fakes():
    r = simulate(PLAN, minutes=3, seed=1)
    gt, tr, trk = r["gt"], r["trajectories"], r["tracklets"]
    vid = gt.groupby("gt_transit")["tracklet_id"].nunique().idxmax()
    tids = gt.loc[gt["gt_transit"] == vid, "tracklet_id"]
    tp = build(vid, tr[tr["tracklet_id"].isin(tids)], trk, Router(PLAN))
    assert tp.observed and tp.t_end > tp.t_start
    ts = np.concatenate([s["t"] for s in tp.observed])
    assert (np.diff(np.unique(ts)) > 0).all()
    mid = (tp.t_start + tp.t_end) / 2
    assert position_at(tp, mid) is not None


def _md():
    from mct.geometry.map_view import MapData
    r = simulate(PLAN, minutes=3, seed=1)
    gt = r["gt"]
    return MapData(PLAN, None, r["fovs"], r["trajectories"], r["tracklets"],
                   gt[["tracklet_id"]].assign(global_id=gt["gt_transit"]), None, "test"), gt


def test_mapdata_paths_for_all_tracks():
    md, gt = _md()
    for gid in md.global_ids()[:15]:
        tp = md.path(gid)
        assert tp.observed, gid


def test_track_figure_renders():
    import pytest
    pytest.importorskip("plotly")
    from mct.geometry.map_view import track_figure
    md, _ = _md()
    gid = md.global_ids()[0]
    fig, tp = track_figure(md, gid, t_now=(md.path(gid).t_start + 1))
    assert len(fig.data) > 3
