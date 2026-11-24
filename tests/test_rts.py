import numpy as np

import motion
from motion import ClickDetector, OneEuro, PinchDetector
from rts.pathing import astar
from rts.world import HOUSE_COST, TILE, World, rect_ring, tile_center


def test_astar_routes_around_wall():
    b = np.zeros((10, 10), bool)
    b[:, 5] = True
    b[8, 5] = False  # gap
    path = astar(b, (1, 1), {(8, 1)})
    assert path and path[0] == (1, 1) and path[-1] == (8, 1)
    assert (5, 8) in path
    assert all(not b[y, x] for x, y in path)


def test_astar_unreachable_and_no_corner_cutting():
    b = np.zeros((5, 5), bool)
    b[:, 2] = True
    assert astar(b, (0, 0), {(4, 0)}) is None
    b = np.zeros((3, 3), bool)
    b[0, 1] = b[1, 0] = True  # diagonal squeeze (0,0)->(1,1)
    assert astar(b, (0, 0), {(1, 1)}) is None


def run(w, seconds, dt=0.05):
    for _ in range(int(seconds / dt)):
        w.update(dt)


def test_gather_loop_earns_gold():
    w = World()
    mine = w.mines[0]
    w.select_box(0, 0, 10_000, 10_000)
    assert len(w.sel_units) == 3
    mx, my = tile_center(mine.rect[:2])
    assert w.command_at(mx, my) == "gather"
    start = w.gold
    run(w, 60)
    assert w.gold > start + 50
    assert mine.gold < 500


def test_move_command_arrives():
    w = World()
    w.select_point(w.units[0].x, w.units[0].y)
    assert len(w.sel_units) == 1
    tx, ty = tile_center((9, 20))
    assert w.command_at(tx, ty) == "move"
    run(w, 10)
    u = w.units[0]
    assert u.state == "idle" and abs(u.x - tx) < TILE and abs(u.y - ty) < TILE


def test_train_worker_supply_and_cost():
    w = World()
    hall = w.buildings[0]
    w.gold = 1000
    assert w.supply_cap == 5 and w.supply_used == 3
    assert w.train_worker(hall) and w.train_worker(hall)
    assert not w.train_worker(hall)  # supply capped
    assert w.gold == 900
    run(w, 12)
    assert len(w.units) == 5


def test_build_house_raises_supply():
    w = World()
    w.select_point(w.units[0].x, w.units[0].y)
    tile = (4, 16)
    assert w.can_place("house", tile)
    g = w.gold
    assert w.place_building("house", tile)
    assert w.gold == g - HOUSE_COST
    run(w, 20)
    assert w.buildings[-1].done and w.supply_cap == 9


def test_cannot_place_on_mine_or_without_gold():
    w = World()
    w.select_point(w.units[0].x, w.units[0].y)
    assert not w.can_place("house", w.mines[0].rect[:2])
    w.gold = 0
    assert not w.place_building("house", (4, 16))


def _hand(index_tip_ext):
    """Synthetic landmarks: other fingers at extension 1.0, index at the given extension."""
    l = np.zeros((21, 3))
    l[9] = (0, 1, 0)  # palm length 1
    l[12] = l[16] = l[20] = (0, 1.0, 0)
    l[8] = (0, index_tip_ext, 0)
    return l


def test_click_detector_tap_not_hold():
    det, t, n = ClickDetector(), 0.0, 0
    seq = [1.9] * 10 + [1.8, 1.6, 1.4, 1.5, 1.7, 1.9] + [1.9] * 10
    for e in seq:
        n += det.update(_hand(e), t)
        t += 1 / 30
    assert n == 1
    det, t, n = ClickDetector(), 0.0, 0
    for e in [1.9] * 10 + [1.5] * 40:
        n += det.update(_hand(e), t)
        t += 1 / 30
    assert n == 0


def test_pinch_hysteresis():
    p = PinchDetector()
    l = np.zeros((21, 3)); l[9] = (0, 1, 0)
    def at(d):
        l[4] = (0, 0, 0); l[8] = (d, 0, 0); return p.update(l)
    assert not at(0.6) and at(0.2) and at(0.35) and not at(0.5)


def test_one_euro_smooths_jitter_keeps_motion():
    f, rng = OneEuro(), np.random.default_rng(0)
    out = [f([0.5 + rng.normal(0, 0.01), 0.5], i / 30) for i in range(60)]
    assert np.std([o[0] for o in out[20:]]) < 0.006
    f = OneEuro()
    for i in range(30):
        y = f([i / 30, 0.5], i / 30)
    assert abs(y[0] - 29 / 30) < 0.08  # little lag on fast motion


def test_region_from_corners_maps_targets_to_margins():
    from calibration import region_from_corners
    p0, p1 = (0.30, 0.20), (0.70, 0.60)
    x0, y0, x1, y1 = region_from_corners(p0, p1)
    fx = lambda p, a, b: (p - a) / (b - a)
    assert abs(fx(p0[0], x0, x1) - 0.06) < 1e-9 and abs(fx(p1[0], x0, x1) - 0.94) < 1e-9
    assert abs(fx(p0[1], y0, y1) - 0.08) < 1e-9 and abs(fx(p1[1], y0, y1) - 0.92) < 1e-9


def test_calibration_flow_pauses_then_captures():
    from calibration import CalibrationFlow, region_from_corners
    f, t = CalibrationFlow(), 0.0
    tl, br = (0.30, 0.20), (0.70, 0.60)

    def feed(k, secs, dt=1 / 30):
        nonlocal t
        out = None
        for _ in range(int(secs / dt)):
            t += dt
            out = f.update(t, k) or out
        return out

    assert feed(tl, 1.0) is None and f.phase == "intro"      # nothing happens before start()
    f.start(t)
    feed((0.9, 0.9), 1.9)                                      # still settling: hand moving, nothing recorded
    assert f.phase == "settle" and not f.points
    feed(tl, 0.3)                                              # settle over -> capturing
    assert f.phase in ("capture", "settle") and len(f.points) <= 1
    feed(tl, 0.8)
    assert len(f.points) == 1 and f.phase == "settle"          # TL captured, pause before BR
    assert feed(br, 3.0) is not None and f.phase == "done"
    assert np.allclose(f.region, region_from_corners(tl, br), atol=0.01)


def test_calibration_flow_retries_when_shaky_and_restarts_when_hand_lost():
    from calibration import CalibrationFlow
    f = CalibrationFlow(); t = 0.0
    f.start(t)
    rng = np.random.default_rng(1)
    for i in range(int(3.0 * 30)):
        t += 1 / 30
        f.update(t, (0.5 + (0.05 if i % 2 else -0.05), 0.5))   # shaky hand
    assert not f.points and f.shaky
    t += 0.1
    f.update(t, None)
    assert f.phase == "settle" and f.progress == 0.0


def test_calibration_anchor_default_and_per_hand(tmp_path, monkeypatch):
    import calibration
    monkeypatch.setattr(calibration, "PATH", tmp_path / "c.json")
    assert calibration.load("Left")["anchor"] == 8
    calibration.save({"region": [0, 0, 1, 1], "anchor": 5}, "Left")
    assert calibration.load("Left")["anchor"] == 5 and calibration.load("Right")["anchor"] == 8


def test_filters_converge_and_smooth(tmp_path, monkeypatch):
    from motion import FILTERS, make_filter
    rng = np.random.default_rng(1)
    for name in FILTERS:
        f = make_filter(name)
        out = [f(np.array([0.5, 0.5]) + rng.normal(0, 0.01, 2), i / 30) for i in range(120)]
        assert np.allclose(out[-1], 0.5, atol=0.03), name
        if name != "none":
            assert np.std([o[0] for o in out[60:]]) < 0.01, name


def test_legacy_calibration_migrates(tmp_path, monkeypatch):
    import json, calibration
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"Left": {"region": [0, 0, 1, 1], "min_cutoff": 3.0, "beta": 5.0}}))
    monkeypatch.setattr(calibration, "PATH", p)
    cal = calibration.load("Left")
    drv = cal["filter_presets"][cal["filter_driver"]]
    assert drv["filter"] == "oneeuro" and drv["params"]["min_cutoff"] == 3.0 and drv["on"]


def test_filter_presets_roundtrip_and_validation(tmp_path, monkeypatch):
    import calibration
    monkeypatch.setattr(calibration, "PATH", tmp_path / "c.json")
    assert [p["filter"] for p in calibration.load("Right")["filter_presets"]] == ["oneeuro", "ema", "kalman"]
    presets = [{"filter": "ema", "params": {"alpha": 0.1}, "on": False}, {"filter": "bogus", "params": {}, "on": True}]
    calibration.update("Right", filter_presets=presets, filter_driver=5)
    cal = calibration.load("Right")
    assert len(cal["filter_presets"]) == 1 and cal["filter_driver"] == 0   # bogus dropped, driver clamped
    assert cal["filter_presets"][0]["on"] is True                          # driver is always plotted
    assert cal["filter_presets"][0]["params"]["alpha"] == 0.1


def test_track_task_completes_and_plots(tmp_path):
    import tracks
    task = tracks.TrackTask(n=3)
    assert task.circles == tracks.TrackTask(n=3).circles  # fixed layout
    t = 0.0
    while not task.finished and t < 60:
        tx, ty = task.target
        t += 0.033
        task.step(t, (tx, ty), (0.5, 0.5), (0.5, 0.5), (tx + 3, ty))
    assert task.finished and len(task.reached) == 3 and task.reached[0] >= 1.0
    d = task.to_dict("Left", 8, (0.2, 0.15, 0.8, 0.75), 0)
    path = tracks.save(d, tmp_path)
    png = tracks.plot(tracks.load(path), tmp_path / "t.png")
    assert png.stat().st_size > 5000
    task2 = tracks.TrackTask(n=2)
    task2.step(0.0, (task2.target[0], task2.target[1]), (0, 0), (0, 0), (0, 0))
    task2.step(0.5, (0, 0), (0, 0), (0, 0), (0, 0))   # leaving the circle resets the dwell
    assert task2.dwell0 is None
