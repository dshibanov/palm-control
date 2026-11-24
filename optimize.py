"""Offline cursor-filter optimizer: replay recorded raw tracks (tracks.py) through every filter.

Each parameter set gets two numbers, in screen px:
  lag     mean distance between the raw point and the filtered point over the whole track
  jitter  wobble of the filtered point while you hold still inside a circle: RMS distance from a straight line
          fitted to each dwell (the last `dwell` seconds before a circle was completed; there the raw wobble
          is pure noise, and the line removes slow drift and the filter still settling)
"No filter" has lag 0 and the highest jitter, a heavy filter the opposite, so we do not minimise one number:

  2. Pareto front     all non-dominated (jitter, lag) trade-offs per filter  -> plot + table
  3. weighted score   argmin lag + LAM * jitter                              (--lam)
  4. jitter budget    argmin lag  subject to  jitter <= BUDGET               (--budget px, or --budget-frac x raw jitter)

  uv run optimize.py                       all tracks in tracks/
  uv run optimize.py tracks/a.json ...     chosen tracks
  uv run optimize.py --apply left          also add the winners to the Left hand's filter presets

Outputs: tracks/opt-<time>.png (plot, opened) and .json (every evaluated point on the fronts + winners).
Caveat: the track was recorded through the live filter, so lag is optimistic for slow filters (you steered to
compensate). Confirm the winners live with the circle task, and prefer 2-3 tracks to avoid overfitting.
"""

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np

import tracks
from motion import FILTERS, make_filter

SEARCH = ("ema", "oneeuro", "kalman")


# ---------------------------------------------------------------- data + metric
class Track:
    def __init__(self, d: dict, name=""):
        s = d["samples"]
        self.name = name
        self.t = np.array(s["t"], float)
        self.scale = np.array([d["screen"][0] - 1, d["screen"][1] - 1], float)
        self.raw = np.array(s["raw"], float)  # 0..1 in the active region: exactly what the filters get live
        self.segs = []  # sample indices of each dwell
        for c in d["circles"]:
            if c["t_reached"] is not None:
                idx = np.where((self.t > c["t_reached"] - d["dwell"]) & (self.t <= c["t_reached"]))[0]
                if len(idx) >= 4:
                    self.segs.append(idx)
        self.n_dwell = sum(len(i) for i in self.segs)

    def run(self, filt) -> np.ndarray:
        return np.array([filt(p, t) for p, t in zip(self.raw, self.t)]) * self.scale

    def metrics(self, filt) -> tuple[float, float, int]:
        out = self.run(filt)
        lag = np.linalg.norm(out - self.raw * self.scale, axis=1)
        rms = []
        for idx in self.segs:
            t, p = self.t[idx] - self.t[idx].mean(), out[idx]
            fit = np.polyfit(t, p, 1)  # one line per axis
            res = p - (np.outer(t, fit[0]) + fit[1])
            rms.append(np.sqrt((res ** 2).sum(axis=1).mean()))
        return float(lag.mean()), float(np.mean(rms)) if rms else float("nan"), self.n_dwell


class Evaluator:
    def __init__(self, data: list[Track]):
        self.data, self.cache = data, {}

    def __call__(self, name, params) -> tuple[float, float]:
        key = (name, tuple(sorted(params.items())))
        if key not in self.cache:
            m = [tr.metrics(make_filter(name, params)) for tr in self.data]
            w = np.array([x[2] for x in m], float)
            lag = float(np.mean([x[0] for x in m]))
            jit = float(np.average([x[1] for x in m], weights=w)) if w.sum() else float("nan")
            self.cache[key] = (lag, jit)
        return self.cache[key]


# ---------------------------------------------------------------- search
def sample(spec, rng):
    out = {}
    for p in spec:
        if p.log:
            out[p.key] = round(math.exp(rng.uniform(math.log(max(p.lo, 1e-3)), math.log(p.hi))), 4)
        else:
            out[p.key] = round(rng.uniform(p.lo, p.hi), 4)
    return out


def perturb(spec, params, rng, scale):
    out = dict(params)
    for p in spec:
        v = params[p.key]
        v = v * math.exp(rng.gauss(0, scale)) if p.log else v + rng.gauss(0, scale * (p.hi - p.lo))
        out[p.key] = round(min(p.hi, max(p.lo, v)), 4)
    return out


def explore(ev: Evaluator, name, n, rng) -> list[tuple[dict, float, float]]:
    spec = FILTERS[name].SPEC
    pts = [{p.key: p.default for p in spec}] + [sample(spec, rng) for _ in range(n)]
    out = []
    for prm in pts:
        lag, jit = ev(name, prm)
        if not (math.isnan(jit) or math.isnan(lag)):
            out.append((prm, lag, jit))
    return out


def refine(ev, name, start, cost, rng, rounds=60):
    """Hill climb on cost(lag, jitter) with a shrinking random step; returns the best (params, lag, jit)."""
    spec = FILTERS[name].SPEC
    best = start
    for i in range(rounds):
        cand = perturb(spec, best[0], rng, 0.3 * (1 - i / rounds) + 0.02)
        lag, jit = ev(name, cand)
        if not math.isnan(jit) and cost(lag, jit) < cost(best[1], best[2]):
            best = (cand, lag, jit)
    return best


def pareto(pts):
    """Non-dominated points (lower lag AND lower jitter is better), sorted by jitter."""
    out, best_lag = [], math.inf
    for p in sorted(pts, key=lambda q: (q[2], q[1])):
        if p[1] < best_lag - 1e-9:
            out.append(p)
            best_lag = p[1]
    return out


# ---------------------------------------------------------------- main
def fmt(params):
    return " ".join(f"{k}={v:g}" for k, v in params.items()) or "-"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="track json files (default: all tracks/track-*.json)")
    ap.add_argument("--lam", type=float, default=8.0, help="approach 3: weight of jitter in lag + lam*jitter (default 8)")
    ap.add_argument("--budget", type=float, help="approach 4: max jitter in px")
    ap.add_argument("--budget-frac", type=float, default=0.4, help="approach 4: budget as a fraction of the raw jitter (default 0.4)")
    ap.add_argument("-n", type=int, default=400, help="random samples per filter (default 400)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--apply", choices=["left", "right"], help="add the winners to this hand's filter presets")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()

    files = [Path(f) for f in a.files] or sorted(tracks.TRACKS_DIR.glob("track-*.json"))
    if not files:
        raise SystemExit("no tracks found: run the circle task first (lab: cursor mode, key T)")
    data = [Track(tracks.load(f), f.name) for f in files]
    ev, rng = Evaluator(data), random.Random(a.seed)
    print(f"{len(files)} track(s): {', '.join(f.name for f in files)}  ({sum(len(t.t) for t in data)} samples,"
          f" {sum(t.n_dwell for t in data)} in dwells)")

    ref_lag, raw_jit = ev("none", {})
    budget = a.budget if a.budget is not None else a.budget_frac * raw_jit
    print(f"raw (no filter): lag {ref_lag:.1f} px, jitter {raw_jit:.2f} px   ->   budget {budget:.2f} px, lam {a.lam:g}\n")

    cost_w = lambda lag, jit: lag + a.lam * jit
    cost_b = lambda lag, jit: lag + 1e3 * max(0.0, jit - budget)  # soft wall so the climb can approach the budget

    res = {}
    for name in SEARCH:
        pts = explore(ev, name, a.n, rng)
        front = pareto(pts)
        w0 = min(pts, key=lambda p: cost_w(p[1], p[2]))
        b0 = min(pts, key=lambda p: cost_b(p[1], p[2]))
        res[name] = {"points": pts, "front": front,
                     "weighted": refine(ev, name, w0, cost_w, rng),
                     "budget": refine(ev, name, b0, cost_b, rng),
                     "default": next(p for p in pts if p[0] == {q.key: q.default for q in FILTERS[name].SPEC})}
        front = pareto(pts + [res[name]["weighted"], res[name]["budget"]])
        res[name]["front"] = front

    rows = [("none", "-", "raw", ref_lag, raw_jit)]
    for name, r in res.items():
        d = r["default"]
        rows.append((name, fmt(d[0]), "default", d[1], d[2]))
        rows.append((name, fmt(r["weighted"][0]), f"3 weighted (lam {a.lam:g})", r["weighted"][1], r["weighted"][2]))
        b = r["budget"]
        ok = "" if b[2] <= budget + 1e-6 else " (budget not reachable)"
        rows.append((name, fmt(b[0]), f"4 budget <= {budget:.2f}{ok}", b[1], b[2]))
    print(f"{'filter':8} {'params':44} {'approach':30} {'lag px':>7} {'jitter':>7}")
    for n, p, ap_, lag, jit in rows:
        print(f"{n:8} {p:44} {ap_:30} {lag:7.1f} {jit:7.2f}")
    print("\nPareto fronts (2): jitter -> lag, per filter")
    for name, r in res.items():
        print(f"  {name}: " + "  ".join(f"[{p[2]:.2f} | {p[1]:.1f}]" for p in r["front"][:: max(1, len(r['front']) // 8)]))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_json = tracks.TRACKS_DIR / f"opt-{stamp}.json"
    out_json.write_text(json.dumps({
        "tracks": [f.name for f in files], "lam": a.lam, "budget": budget, "raw": {"lag": ref_lag, "jitter": raw_jit},
        "filters": {n: {"default": r["default"], "weighted": r["weighted"], "budget": r["budget"], "front": r["front"]}
                    for n, r in res.items()}}, indent=1))
    png = plot(res, ref_lag, raw_jit, budget, a.lam, out_json.with_suffix(".png"))
    print(f"\nwrote {out_json.name} and {png.name}")
    if not a.no_open:
        tracks.open_file(png)
    if a.apply:
        apply(a.apply.capitalize(), res, raw_jit, budget)


def plot(res, ref_lag, raw_jit, budget, lam, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = {"ema": "#4aa3ff", "oneeuro": "#ffb43c", "kalman": "#58d68d"}
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=100)
    ax.axvline(budget, color="#c0392b", ls="--", lw=1, label=f"4: jitter budget {budget:.2f}")
    for name, r in res.items():
        xs, ys = [p[2] for p in r["points"]], [p[1] for p in r["points"]]
        ax.scatter(xs, ys, s=7, color=cols[name], alpha=0.25)
        f = r["front"]
        ax.plot([p[2] for p in f], [p[1] for p in f], "-", color=cols[name], lw=2, label=f"2: {name} Pareto front")
        d, w, b = r["default"], r["weighted"], r["budget"]
        ax.scatter(d[2], d[1], marker="s", s=60, color=cols[name], ec="k", zorder=5)
        ax.scatter(w[2], w[1], marker="*", s=170, color=cols[name], ec="k", zorder=6)
        ax.scatter(b[2], b[1], marker="D", s=70, color=cols[name], ec="k", zorder=6)
    ax.scatter(raw_jit, ref_lag, marker="o", s=80, color="k", zorder=7, label="no filter")
    for m, lab in (("s", "default params"), ("*", f"3: best lag + {lam:g}*jitter"), ("D", "4: best within budget")):
        ax.scatter([], [], marker=m, s=60, color="#999", ec="k", label=lab)
    ax.set_xlabel("jitter while holding still  (px RMS, lower = steadier)")
    ax.set_ylabel("lag  (mean distance raw -> filtered, px, lower = more responsive)")
    ax.set_title("Cursor filters: every dot is one parameter set; lines are the best trade-offs")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(png)
    plt.close(fig)
    return png


def apply(hand, res, raw_jit, budget):
    import calibration
    cal = calibration.load(hand)
    presets = cal["filter_presets"]
    added = 0
    for name, r in res.items():
        for key in ("budget", "weighted"):
            prm, _, jit = r[key]
            if jit > 0.95 * raw_jit or (key == "budget" and jit > budget + 1e-6):
                continue  # barely smooths anything / misses the budget: not worth a preset slot
            if any(p["filter"] == name and p["params"] == prm for p in presets):
                continue
            if len(presets) >= calibration.MAX_PRESETS:
                print(f"preset list is full ({calibration.MAX_PRESETS}); remove some in the lab, then re-run --apply")
                break
            presets.append({"filter": name, "params": prm, "on": True})
            added += 1
    calibration.update(hand, filter_presets=presets)
    print(f"added {added} preset(s) to the {hand} hand (restart the lab to see them)")


if __name__ == "__main__":
    main()
