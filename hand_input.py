"""Hand input layer: camera + MediaPipe in a background thread.

Produces, per detected hand: filtered cursor, pinch state, built-in/custom gesture.
And an event queue: click, rclick, pinch_down, pinch_up, gesture.

Cursor follows one landmark (`anchor`, default 5 = index *knuckle*; changeable in the cursor lab).
The knuckle is the default because the fingertip moves when you tap. The cursor is also
frozen while a tap is in progress.
"""

import threading
import time
from collections import deque, namedtuple
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import calibration
from gestures import GestureStore
from motion import ClickDetector, OneEuro, PinchDetector

MODEL = Path(__file__).parent / "models" / "gesture_recognizer.task"
CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]
Event = namedtuple("Event", "kind hand name")


@dataclass
class Hand:
    label: str
    lms: np.ndarray
    builtin: str | None
    score: float
    custom: str | None
    cursor: tuple[float, float]   # 0..1 inside the active region (mirrored camera space)
    pinch: bool
    palm: tuple[float, float]     # landmark 9, normalized camera coords
    point: tuple[float, float] = (0.0, 0.0)  # the tracked anchor landmark, normalized camera coords


@dataclass
class Snapshot:
    hands: dict = field(default_factory=dict)  # label -> Hand
    frame: np.ndarray | None = None            # BGR preview with skeleton drawn
    fps: float = 0.0


class _Track:
    def __init__(self, drop):
        self.click = ClickDetector(drop=drop)
        self.rclick = ClickDetector(drop=drop, finger="middle")
        self.pinch = PinchDetector()
        self.filt = OneEuro()
        self.cursor = (0.5, 0.5)
        self.pinch_changed = -1e9
        self.cand, self.cand_n, self.stable = None, 0, None

    def reset(self):
        self.click.reset(); self.rclick.reset(); self.filt.reset()
        self.cand, self.cand_n, self.stable = None, 0, None


class HandInput:
    def __init__(self, camera=2, click_drop=None, region=None, width=640, height=480,
                 stable_frames=5, min_score=0.6, hand="Right"):
        cal = calibration.load(hand)
        self.region = tuple(region or cal["region"])
        self.min_cutoff, self.beta, self.anchor = cal["min_cutoff"], cal["beta"], cal["anchor"]
        self.camera = camera
        self.click_drop = cal["click_drop"] if click_drop is None else click_drop  # may be changed live
        self.width, self.height = width, height
        self.stable_frames, self.min_score = stable_frames, min_score
        self.store = GestureStore()
        self._snap = Snapshot()
        self._events: deque = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.error: str | None = None

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def snapshot(self) -> Snapshot:
        with self._lock:
            return self._snap

    def drain_events(self) -> list[Event]:
        with self._lock:
            ev = list(self._events)
            self._events.clear()
        return ev

    # ---- internals ----
    def _emit(self, kind, hand, name=None):
        with self._lock:
            self._events.append(Event(kind, hand, name))

    def map_cursor(self, nx, ny):
        x0, y0, x1, y1 = self.region
        return (min(max((nx - x0) / (x1 - x0), 0.0), 1.0), min(max((ny - y0) / (y1 - y0), 0.0), 1.0))

    def _run(self):
        cap = cv2.VideoCapture(self.camera, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not cap.isOpened():
            self.error = f"Cannot open camera {self.camera}"
            return
        rec = vision.GestureRecognizer.create_from_options(
            vision.GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path=str(MODEL)),
                running_mode=vision.RunningMode.VIDEO, num_hands=2))
        tracks: dict[str, _Track] = {}
        t0 = time.monotonic()
        last = t0
        fps = 0.0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.error = "camera read failed"
                    break
                frame = cv2.flip(frame, 1)
                now = time.monotonic()
                fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
                last = now
                res = rec.recognize_for_video(
                    mp.Image(mp.ImageFormat.SRGB, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), int((now - t0) * 1000))
                hands = {}
                for i, (lm, handed) in enumerate(zip(res.hand_landmarks, res.handedness)):
                    label = handed[0].category_name
                    arr = np.array([[p.x, p.y, p.z] for p in lm])
                    g = res.gestures[i][0] if i < len(res.gestures) and res.gestures[i] else None
                    builtin = g.category_name if g and g.category_name != "None" else None
                    score = g.score if g else 0.0
                    tr = tracks.setdefault(label, _Track(self.click_drop))
                    hands[label] = self._update_hand(label, arr, builtin, score, tr, now)
                    self._draw(frame, arr)
                for label in list(tracks):
                    if label not in hands:
                        tracks.pop(label).reset()
                small = cv2.resize(frame, (192, 144))
                with self._lock:
                    self._snap = Snapshot(hands, small, fps)
        finally:
            cap.release()
            rec.close()

    def _update_hand(self, label, arr, builtin, score, tr: _Track, now) -> Hand:
        # pinch
        was = tr.pinch.down
        pinch = tr.pinch.update(arr)
        if pinch != was:
            tr.pinch_changed = now
            self._emit("pinch_down" if pinch else "pinch_up", label)
        # taps (ignored around a pinch, which also bends the index finger)
        tr.click.drop = tr.rclick.drop = self.click_drop
        clicked = tr.click.update(arr, now)
        rclicked = tr.rclick.update(arr, now)
        if not pinch and now - tr.pinch_changed > 0.4:
            if clicked:
                self._emit("click", label)
            elif rclicked:
                self._emit("rclick", label)
        # cursor (frozen during a tap)
        raw = self.map_cursor(arr[self.anchor, 0], arr[self.anchor, 1])
        tr.filt.min_cutoff, tr.filt.beta = self.min_cutoff, self.beta
        if not (tr.click.busy(now) or tr.rclick.busy(now)):
            tr.cursor = tuple(tr.filt(raw, now))
        # stable built-in gesture -> event on change
        name = builtin if builtin and score >= self.min_score else None
        if name == tr.cand:
            tr.cand_n += 1
        else:
            tr.cand, tr.cand_n = name, 1
        if tr.cand_n >= self.stable_frames and tr.cand != tr.stable:
            tr.stable = tr.cand
            if tr.stable:
                self._emit("gesture", label, tr.stable)
        custom, _ = self.store.classify(arr, label)
        return Hand(label, arr, builtin, score, custom, tr.cursor, pinch, (float(arr[9, 0]), float(arr[9, 1])),
                    (float(arr[self.anchor, 0]), float(arr[self.anchor, 1])))

    @staticmethod
    def _draw(frame, arr):
        h, w = frame.shape[:2]
        pts = [(int(p[0] * w), int(p[1] * h)) for p in arr]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (80, 220, 120), 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, p, 3, (255, 255, 255), -1, cv2.LINE_AA)
