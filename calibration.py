"""Persistent hand-control calibration (calibration.json)."""

import json
from pathlib import Path

import numpy as np

PATH = Path(__file__).parent / "calibration.json"
DEFAULT = {"region": [0.2, 0.15, 0.8, 0.75], "min_cutoff": 1.5, "beta": 10.0, "anchor": 5,  # anchor = landmark index
           "click_drop": 0.2}  # finger dip needed for a tap (lower = more sensitive)


def _read() -> dict:
    return json.loads(PATH.read_text()) if PATH.exists() else {}


def load(hand: str = "Right") -> dict:
    """Calibration for one hand ("Left"/"Right"); falls back to defaults."""
    cal = dict(DEFAULT)
    cal.update(_read().get(hand, {}))
    return cal


def save(cal: dict, hand: str = "Right"):
    data = _read()
    data[hand] = cal
    PATH.write_text(json.dumps(data, indent=2))


def region_from_corners(p0, p1, margin=(0.06, 0.08)):
    """Camera region (x0, y0, x1, y1) so that knuckle position p0 maps to screen fraction `margin`
    and p1 maps to 1 - margin. p0/p1 are knuckle positions (normalized camera coords) held at the
    top-left / bottom-right target."""
    out = []
    for a, b, m in zip(p0, p1, margin):
        span = (b - a) / (1 - 2 * m)
        lo = a - m * span
        out += [lo, lo + span]
    return (out[0], out[2], out[1], out[3])


class CalibrationFlow:
    """Two-corner calibration with pauses: intro -> (settle -> capture) x2 -> done.

    intro   explanation screen, waits for start()
    settle  SETTLE_S seconds to move the hand to the target and let the cursor stabilize (nothing is recorded;
            the timer restarts if the hand disappears)
    capture CAPTURE_S seconds of samples, accepted only if the hand held still (else retried)
    """

    SETTLE_S, CAPTURE_S, STILL_STD, MIN_SAMPLES = 2.0, 0.6, 0.015, 8

    def __init__(self):
        self.phase, self.step, self.points, self.shaky = "intro", 0, [], False
        self.t0, self.samples, self.region, self.elapsed = 0.0, [], None, 0.0
        self.last_std = 0.0  # live movement (max std of x/y, normalized camera coords) during capture

    def start(self, now: float):
        self.phase, self.step, self.points, self.shaky = "settle", 0, [], False
        self.t0, self.samples = now, []

    @property
    def progress(self) -> float:
        if self.phase not in ("settle", "capture"):
            return 0.0
        return min(self.elapsed / (self.SETTLE_S if self.phase == "settle" else self.CAPTURE_S), 1.0)

    def update(self, now: float, knuckle):
        """Feed one frame (knuckle = (x, y) normalized, or None if no hand). Returns the region when finished."""
        if self.phase in ("intro", "done"):
            return None
        self.elapsed = now - self.t0
        if knuckle is None:  # lost the hand: start waiting again
            self.t0, self.samples, self.elapsed = now, [], 0.0
            if self.phase == "capture":
                self.phase = "settle"
            return None
        if self.phase == "settle":
            if self.elapsed >= self.SETTLE_S:
                self.phase, self.t0, self.samples, self.elapsed, self.shaky = "capture", now, [], 0.0, False
            return None
        self.samples.append(tuple(knuckle))
        pts = np.array(self.samples)
        self.last_std = float(pts.std(axis=0).max()) if len(pts) > 2 else 0.0
        if self.elapsed < self.CAPTURE_S:
            return None
        if len(pts) >= self.MIN_SAMPLES and pts.std(axis=0).max() < self.STILL_STD:
            self.points.append(tuple(pts.mean(axis=0)))
            self.step += 1
            if self.step == 2:
                self.phase, self.region = "done", region_from_corners(*self.points)
                return self.region
            self.phase, self.t0, self.samples, self.elapsed = "settle", now, [], 0.0
        else:  # moved during capture: retry the capture window
            self.shaky, self.t0, self.samples, self.elapsed = True, now, [], 0.0
        return None
