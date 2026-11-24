"""Movement gestures: finger "click" (dip down and back), pinch, cursor smoothing.

Click signal = a finger's extension minus the mean extension of the other
three fingers. Extension = |tip - wrist| / |middle MCP - wrist|, so it is
invariant to hand position and size, and whole-hand movement cancels out.
A click is: signal drops by >= `drop`, then recovers to near its starting
level, all within `window` seconds.
"""

import math
from collections import deque

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


class OneEuro:
    """One Euro filter for 2D points: smooth when slow, responsive when fast."""

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
