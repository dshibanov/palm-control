"""Movement gestures: finger "click" (dip down and back), pinch, cursor smoothing.

Click signal = a finger's extension minus the mean extension of the other
three fingers. Extension = |tip - wrist| / |middle MCP - wrist|, so it is
invariant to hand position and size, and whole-hand movement cancels out.
A click is: signal drops by >= `drop`, then recovers to near its starting
level, all within `window` seconds.
"""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}


def _palm(lms):
    return np.linalg.norm(lms[9, :2] - lms[0, :2]) or 1.0


def _ext(lms, tip):
    return np.linalg.norm(lms[tip, :2] - lms[0, :2]) / _palm(lms)


def click_signal(lms: np.ndarray, finger: str = "index") -> float:
    others = [t for f, t in TIPS.items() if f != finger]
    return _ext(lms, TIPS[finger]) - float(np.mean([_ext(lms, t) for t in others]))


class ClickDetector:
    def __init__(self, drop=0.2, window=0.8, cooldown=0.4, recover=0.7, finger="index"):
        self.drop, self.window, self.cooldown, self.recover = drop, window, cooldown, recover
        self.finger = finger
        self.hist: deque = deque()  # (t, smoothed signal)
        self.trace: deque = deque(maxlen=90)  # for plotting
        self.last_click = -1e9
        self.dipping = False
        self._ema = None

    def busy(self, t: float) -> bool:
        """True while a click is in progress or just happened (used to freeze the cursor)."""
        return self.dipping or t - self.last_click < 0.2

    def update(self, lms: np.ndarray, t: float) -> bool:
        s = click_signal(lms, self.finger)
        self._ema = s if self._ema is None else 0.5 * self._ema + 0.5 * s
        self.hist.append((t, self._ema))
        self.trace.append(self._ema)
        while self.hist and t - self.hist[0][0] > self.window:
            self.hist.popleft()
        if len(self.hist) < 4:
            self.dipping = False
            return False
        vals = np.array([v for _, v in self.hist])
        self.dipping = (vals.max() - vals[-1]) >= 0.4 * self.drop and int(vals.argmax()) < len(vals) - 1
        if t - self.last_click < self.cooldown:
            return False
        i_min = int(vals.argmin())
        if i_min == 0 or i_min == len(vals) - 1:
            return False
        start = vals[:i_min].max()
        depth = start - vals[i_min]
        if depth >= self.drop and vals[-1] >= vals[i_min] + self.recover * depth:
            self.last_click = t
            self.hist.clear()
            self.dipping = False
            return True
        return False

    def reset(self):
        self.hist.clear()
        self.dipping = False
        self._ema = None


class PinchDetector:
    """Thumb tip - index tip distance (in palm lengths) with hysteresis."""

    def __init__(self, on=0.25, off=0.45):
        self.on, self.off, self.down = on, off, False

    def update(self, lms: np.ndarray) -> bool:
        d = np.linalg.norm(lms[4, :2] - lms[8, :2]) / _palm(lms)
        if self.down and d > self.off:
            self.down = False
        elif not self.down and d < self.on:
            self.down = True
        return self.down


@dataclass(frozen=True)
class Param:
    """A tunable filter parameter. log=True steps multiplicatively (x step), else additively (+ step)."""
    key: str
    label: str
    default: float
    lo: float
    hi: float
    step: float
    log: bool = False
    hint: str = ""

    def bump(self, v: float, direction: int) -> float:
        v = v * self.step ** direction if self.log else v + direction * self.step
        return round(min(self.hi, max(self.lo, v)), 4)


class Identity:
    """No filtering: the cursor is exactly the raw point."""
    SPEC: list = []

    def __init__(self):
        pass

    def __call__(self, x, t):
        return np.asarray(x, float)

    def reset(self):
        pass


class EMA:
    """Exponential moving average: x = alpha*new + (1-alpha)*x. Simple; same lag at any speed."""
    SPEC = [Param("alpha", "alpha", 0.35, 0.02, 1.0, 0.05, hint="1 = no smoothing, lower = smoother but laggier")]

    def __init__(self, alpha=0.35):
        self.alpha, self.x = alpha, None

    def __call__(self, x, t):
        x = np.asarray(x, float)
        self.x = x if self.x is None else self.alpha * x + (1 - self.alpha) * self.x
        return self.x

    def reset(self):
        self.x = None


class Kalman:
    """Constant-velocity Kalman filter per axis. Predicts where the point is going, so it lags less than
    an EMA of the same smoothness, but overshoots a bit when you stop suddenly.
    q = how erratically the hand may accelerate, r = how noisy the tracking is (in screen fractions)."""
    SPEC = [Param("q", "q motion", 30.0, 0.5, 2000.0, 1.5, True, "higher = trusts motion less, follows faster"),
            Param("r", "r noise", 0.01, 0.0005, 0.1, 1.4, True, "higher = smoother, laggier")]

    def __init__(self, q=30.0, r=0.01):
        self.q, self.r = q, r
        self.reset()

    def reset(self):
        self.s = None  # per axis [pos, vel, p00, p01, p11]
        self.t = None

    def __call__(self, x, t):
        x = np.asarray(x, float)
        if self.s is None:
            self.s = [[float(v), 0.0, 1.0, 0.0, 1.0] for v in x]
            self.t = t
            return x
        dt = max(t - self.t, 1e-3)
        self.t = t
        out = []
        for z, st in zip(x, self.s):
            p, v, p00, p01, p11 = st
            p += v * dt  # predict
            p00, p01, p11 = (p00 + 2 * dt * p01 + dt * dt * p11 + self.q * dt ** 4 / 4,
                             p01 + dt * p11 + self.q * dt ** 3 / 2,
                             p11 + self.q * dt ** 2)
            sv = p00 + self.r ** 2  # update
            k0, k1 = p00 / sv, p01 / sv
            y = z - p
            p, v = p + k0 * y, v + k1 * y
            p00, p01, p11 = (1 - k0) * p00, (1 - k0) * p01, p11 - k1 * p01
            st[:] = [p, v, p00, p01, p11]
            out.append(p)
        return np.array(out)


class OneEuro:
    """One Euro filter for 2D points: smooth when slow, responsive when fast."""
    SPEC = [Param("min_cutoff", "min cutoff", 1.5, 0.1, 10.0, 1.25, True, "lower = steadier when still, laggier"),
            Param("beta", "beta", 10.0, 0.0, 200.0, 1.5, True, "higher = less lag when moving fast"),
            Param("d_cutoff", "speed cutoff", 1.0, 0.1, 10.0, 1.25, True, "smooths the speed estimate; rarely needs a change")]

    def __init__(self, min_cutoff=1.5, beta=10.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, float)
        if self.x is None:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x

    def reset(self):
        self.x = self.dx = self.t = None


FILTERS = {"none": Identity, "ema": EMA, "oneeuro": OneEuro, "kalman": Kalman}
FILTER_INFO = {
    "none": "Raw point, no smoothing.",
    "ema": "Fixed blend with the previous value. Simple, but lag and smoothness are tied together.",
    "oneeuro": "Heavy smoothing when slow, light when fast. Best default for pointing.",
    "kalman": "Predicts motion, so less lag for the same smoothness; may overshoot on sudden stops.",
}


def default_params(name: str) -> dict:
    return {p.key: p.default for p in FILTERS[name].SPEC}


def make_filter(name: str, params: dict | None = None):
    return FILTERS[name](**{**default_params(name), **(params or {})})
