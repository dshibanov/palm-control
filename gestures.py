"""Custom gesture learning: k-NN over normalized hand landmarks.

A sample is the 21x3 landmark array made position- and scale-invariant
(wrist at origin, scaled by wrist->middle-MCP length). Left hands are mirrored
so one set of samples works for either hand. Orientation is kept on purpose,
so e.g. thumbs-up and thumbs-down stay distinguishable.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(__file__).parent / "gestures.json"


def features(lms: np.ndarray, label: str) -> np.ndarray:
    pts = lms - lms[0]
    if label == "Left":
        pts = pts * np.array([-1.0, 1.0, 1.0])
    scale = np.linalg.norm(pts[9, :2]) or 1.0
    return (pts / scale).ravel()


class GestureStore:
    def __init__(self, path: Path = DEFAULT_PATH, k: int = 5, reject: float = 0.7):
        self.path, self.k, self.reject = path, k, reject
        self.samples: dict[str, list[np.ndarray]] = {}
        self._X = self._y = None
        self.load()

    def load(self):
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.samples = {n: [np.array(s) for s in v] for n, v in data.items()}
        self._rebuild()

    def save(self):
        self.path.write_text(json.dumps({n: [s.tolist() for s in v] for n, v in self.samples.items()}))

    def _rebuild(self):
        names = [n for n, v in self.samples.items() for _ in v]
        if names:
            self._X = np.stack([s for v in self.samples.values() for s in v])
            self._y = np.array(names)
        else:
            self._X = self._y = None

    def names(self) -> list[str]:
        return list(self.samples)

    def add_gesture(self, name: str):
        self.samples.setdefault(name, [])

    def add_sample(self, name: str, lms: np.ndarray, label: str):
        self.samples.setdefault(name, []).append(features(lms, label))
        self._rebuild()

    def delete(self, name: str):
        self.samples.pop(name, None)
        self._rebuild()
        self.save()

    def classify(self, lms: np.ndarray, label: str):
        """Return (name, distance) or (None, distance) if nothing is close enough."""
        if self._X is None:
            return None, float("inf")
        d = np.linalg.norm(self._X - features(lms, label), axis=1)
        idx = np.argsort(d)[: self.k]
        winner = Counter(self._y[idx]).most_common(1)[0][0]
        dist = float(d[idx][self._y[idx] == winner].mean())
        return (winner if dist <= self.reject else None), dist
