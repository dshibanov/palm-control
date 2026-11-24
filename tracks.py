"""Circle-reaching test problem: record raw hand tracks to tune cursor filters offline.

The task: numbered circles appear one at a time; reach one with the cursor and stay inside for `dwell`
seconds and the next appears. While it runs, the *raw* (unfiltered) tracked point is recorded for every
camera frame. A finished run is saved as tracks/track-<time>.json and plotted (trajectory + circles).

The same fixed circle layout is used for every run (seed), so tracks are comparable.

  uv run tracks.py plot tracks/track-XXXX.json [--open]     re-plot a saved track
"""

import argparse
import json
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

TRACKS_DIR = Path(__file__).parent / "tracks"


def make_circles(n=12, seed=1, area=(90, 140, 1010, 360), min_gap=300) -> list[tuple[int, int]]:
    """Deterministic circle centres inside area=(x0, y0, x1, y1); each is >= `min_gap` px from the previous one."""
    rng = random.Random(seed)
    x0, y0, x1, y1 = area
    out, prev = [], ((x0 + x1) // 2, (y0 + y1) // 2)
    while len(out) < n:
        p = (rng.randint(x0, x1), rng.randint(y0, y1))
        if np.hypot(p[0] - prev[0], p[1] - prev[1]) >= min_gap:
            out.append(p)
            prev = p
    return out


class TrackTask:
    """Circle sequence + raw-track recorder. Feed it one call per new camera frame via `step`."""

    def __init__(self, n=12, seed=1, size=(1100, 720), radius=42, dwell=1.0, area=(90, 140, 1010, 360)):
        """area: where circles may appear (the main screen only, clear of the text, preview and plot)."""
        self.size, self.radius, self.dwell, self.seed, self.area = size, radius, dwell, seed, area
        self.circles = make_circles(n, seed, area)
        self.idx = 0                  # circle being reached
        self.t0 = None                # wall time of the first frame
        self.dwell0 = None
        self.shown = [0.0]            # time each circle appeared (s since start)
        self.reached = []             # time each circle was completed
        self.rows: list[tuple] = []   # (t, cam_x, cam_y, raw_x, raw_y, rawpx_x, rawpx_y, cur_x, cur_y)

    @property
    def n(self):
        return len(self.circles)

    @property
    def finished(self):
        return self.idx >= self.n

    @property
    def target(self):
        return None if self.finished else self.circles[self.idx]

    def progress(self, now) -> float:
        return 0.0 if self.dwell0 is None else min((now - self.dwell0) / self.dwell, 1.0)

    def step(self, now, cursor_px, cam, raw, raw_px) -> bool:
        """One camera frame with the hand visible. cursor_px = filtered cursor (decides 'reached'),
        cam = tracked point in camera coords, raw = region-mapped raw point (0..1), raw_px = raw in screen px.
        Returns True when the last circle is completed."""
        if self.finished:
            return True
        if self.t0 is None:
            self.t0 = now
        t = now - self.t0
        self.rows.append((t, *cam, *raw, *raw_px, *cursor_px))
        tx, ty = self.target
        if np.hypot(cursor_px[0] - tx, cursor_px[1] - ty) < self.radius:
            self.dwell0 = self.dwell0 if self.dwell0 is not None else now
            if now - self.dwell0 >= self.dwell:
                self.reached.append(t)
                self.idx += 1
                self.dwell0 = None
                if not self.finished:
                    self.shown.append(t)
        else:
            self.dwell0 = None
        return self.finished

    def lost_hand(self):
        self.dwell0 = None

    def to_dict(self, hand, anchor, region, driver) -> dict:
        a = np.array(self.rows) if self.rows else np.zeros((0, 9))
        return {
            "version": 1, "created": datetime.now().isoformat(timespec="seconds"),
            "hand": hand, "anchor": anchor, "region": list(region), "screen": list(self.size),
            "radius": self.radius, "dwell": self.dwell, "seed": self.seed, "driver": driver,
            "circles": [{"i": i + 1, "x": x, "y": y, "t_shown": self.shown[i] if i < len(self.shown) else None,
                         "t_reached": self.reached[i] if i < len(self.reached) else None}
                        for i, (x, y) in enumerate(self.circles)],
            "samples": {"t": a[:, 0].tolist(), "cam": a[:, 1:3].tolist(), "raw": a[:, 3:5].tolist(),
                        "raw_px": a[:, 5:7].tolist(), "cursor_px": a[:, 7:9].tolist()},
        }


def save(track: dict, directory: Path = TRACKS_DIR) -> Path:
    directory.mkdir(exist_ok=True)
    path = directory / f"track-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(track))
    return path


def load(path) -> dict:
    return json.loads(Path(path).read_text())


def plot(track: dict, png: Path) -> Path:
    """Raw trajectory (coloured by time) over numbered circles, in screen coordinates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle

    w, h = track["screen"]
    s = track["samples"]
    t, xy = np.array(s["t"]), np.array(s["raw_px"])
    fig, ax = plt.subplots(figsize=(w / 80 + 1, h / 80), dpi=100)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_facecolor("#14161a")
    cs = track["circles"]
    ax.plot([c["x"] for c in cs], [c["y"] for c in cs], "--", color="#555a66", lw=1, zorder=1)
    for c in cs:
        ax.add_patch(Circle((c["x"], c["y"]), track["radius"], fill=False, ec="#ffb43c", lw=2, zorder=2))
        ax.text(c["x"], c["y"], str(c["i"]), color="#ffb43c", ha="center", va="center", fontsize=12, weight="bold", zorder=3)
    if len(xy) > 1:
        segs = np.stack([xy[:-1], xy[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis", norm=plt.Normalize(t.min(), t.max()), lw=1.6, zorder=4)
        lc.set_array(t[:-1])
        ax.add_collection(lc)
        ax.scatter(xy[:, 0], xy[:, 1], c=t, cmap="viridis", s=5, zorder=5)
        ax.scatter(*xy[0], marker="o", s=70, c="white", ec="black", zorder=6)
        fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.02).set_label("time (s)")
    dur = t.max() if len(t) else 0
    ax.set_title(f"{track['hand']} hand, anchor {track['anchor']}  -  {len(cs)} circles, {dur:.1f} s, {len(t)} raw samples"
                 f"  (white dot = start)")
    for sp in ax.spines.values():
        sp.set_color("#555a66")
    fig.tight_layout()
    fig.savefig(png, facecolor="white")
    plt.close(fig)
    return Path(png)


def open_file(path):
    subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def plot_and_open_async(json_path: Path):
    """Plot in a child process (matplotlib import is slow) and open the PNG when it is ready."""
    subprocess.Popen([sys.executable, str(Path(__file__)), "plot", str(json_path), "--open"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plot")
    p.add_argument("file")
    p.add_argument("--open", action="store_true")
    a = ap.parse_args()
    png = plot(load(a.file), Path(a.file).with_suffix(".png"))
    print(png)
    if a.open:
        open_file(png)
