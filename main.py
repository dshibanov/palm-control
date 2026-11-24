"""Palm Control prototype.

Top half: live webcam with hand landmarks.
Bottom half: landmark values per hand, refreshed every N ms (default 1000),
with the movement (delta) since the previous refresh.

Gestures: MediaPipe's built-in recognizer (Thumb_Up, Victory, ...) plus custom
gestures you teach it (k-NN on landmarks, saved to gestures.json).

Movement: index-finger "click" (dip down and back) is detected per hand; a live
graph of the finger signal is drawn at the bottom of the video.

Keys: n = new custom gesture (type name, Enter)   Tab = select gesture
      space = start/stop recording samples        x = delete selected gesture
      m = toggle real mouse click on index-click (default off, --mouse to start on)
      q / Esc = quit
"""

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from gestures import GestureStore
from motion import ClickDetector
from pynput.mouse import Button, Controller as Mouse

MODEL = Path(__file__).parent / "models" / "gesture_recognizer.task"
CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]
NAMES = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]
HAND_COLORS = {"Left": (255, 160, 60), "Right": (80, 220, 120)}  # BGR
PANEL_H = 350
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_hands(frame, result):
    h, w = frame.shape[:2]
    for lms, handed in zip(result.hand_landmarks, result.handedness):
        label = handed[0].category_name
        color = HAND_COLORS.get(label, (255, 255, 255))
        pts = [(int(p.x * w), int(p.y * h)) for p in lms]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, p, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, label, (pts[0][0] - 20, pts[0][1] + 30), FONT, 0.8, color, 2, cv2.LINE_AA)


def draw_labels(frame, result, store):
    """Built-in + custom gesture label under each hand."""
    h, w = frame.shape[:2]
    out = {}
    for i, (lms, handed) in enumerate(zip(result.hand_landmarks, result.handedness)):
        label = handed[0].category_name
        arr = np.array([[p.x, p.y, p.z] for p in lms])
        builtin = result.gestures[i][0] if i < len(result.gestures) and result.gestures[i] else None
        custom, dist = store.classify(arr, label)
        out[label] = (arr, custom)
        x, y = int(lms[0].x * w) - 20, int(lms[0].y * h) + 56
        lines = [
            (f"builtin: {builtin.category_name} {builtin.score:.2f}" if builtin and builtin.category_name != "None" else "builtin: -", (230, 230, 230)),
            (f"custom: {custom} (d={dist:.2f})" if custom else f"custom: - (d={dist:.2f})" if np.isfinite(dist) else "custom: -", (80, 220, 255)),
        ]
        for j, (t, c) in enumerate(lines):
            cv2.putText(frame, t, (x, y + j * 22), FONT, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, t, (x, y + j * 22), FONT, 0.6, c, 1, cv2.LINE_AA)
    return out


def draw_click_graph(frame, det, clicks, flash, mouse_on):
    """Index-finger signal trace (bottom-left of video) + click counter."""
    h = frame.shape[0]
    x0, y0, gw, gh = 8, h - 80, 180, 70
    cv2.rectangle(frame, (x0, y0), (x0 + gw, y0 + gh), (0, 0, 0), -1)
    if len(det.trace) > 1:
        v = np.array(det.trace)
        pts = [(x0 + int(i * gw / 90), y0 + gh - int(np.clip((a + 0.6) / 1.8, 0, 1) * gh)) for i, a in enumerate(v)]
        cv2.polylines(frame, [np.array(pts)], False, (80, 220, 255), 1, cv2.LINE_AA)
    txt = f"index click  x{clicks}"
    cv2.putText(frame, txt, (x0 + 4, y0 + 14), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    tag = "MOUSE ON" if mouse_on else "mouse off (m)"
    cv2.putText(frame, tag, (x0 + 4, y0 + gh - 6), FONT, 0.45, (0, 0, 255) if mouse_on else (160, 160, 160), 1, cv2.LINE_AA)
    if flash:
        cv2.putText(frame, "CLICK!", (frame.shape[1] // 2 - 80, 60), FONT, 1.6, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(frame, "CLICK!", (frame.shape[1] // 2 - 80, 60), FONT, 1.6, (0, 255, 255), 2, cv2.LINE_AA)


def draw_hud(frame, store, selected, recording, typing):
    names = store.names()
    lines = []
    if typing is not None:
        lines.append((f"new gesture name: {typing}_", (0, 255, 255)))
    for n in names:
        mark = ">" if n == selected else " "
        rec = "  [REC]" if recording and n == selected else ""
        lines.append((f"{mark} {n}  ({len(store.samples[n])}){rec}", (0, 0, 255) if rec else (255, 255, 255)))
    if not names and typing is None:
        lines.append(("press n to teach a gesture", (180, 180, 180)))
    for j, (t, c) in enumerate(lines):
        cv2.putText(frame, t, (8, 20 + j * 20), FONT, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, t, (8, 20 + j * 20), FONT, 0.55, c, 1, cv2.LINE_AA)


def snapshot(result):
    """Latest landmarks as {label: ndarray(21, 3)}."""
    snap = {}
    for lms, handed in zip(result.hand_landmarks, result.handedness):
        snap[handed[0].category_name] = np.array([[p.x, p.y, p.z] for p in lms])
    return snap


def render_panel(width, snap, prev, interval_ms, age_ms):
    panel = np.full((PANEL_H, width, 3), 24, np.uint8)
    cv2.putText(
        panel, f"landmarks snapshot  every {interval_ms} ms  (age {age_ms:4d} ms)",
        (8, 16), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
    )
    col_w = width // 2
    for i, label in enumerate(("Left", "Right")):
        x0 = i * col_w + 8
        color = HAND_COLORS[label]
        cur = snap.get(label)
        if cur is None:
            cv2.putText(panel, f"{label}: no hand", (x0, 40), FONT, 0.5, (110, 110, 110), 1, cv2.LINE_AA)
            continue
        old = prev.get(label)
        cv2.putText(panel, f"{label}   name        x     y     z    move", (x0, 40), FONT, 0.38, color, 1, cv2.LINE_AA)
        for j, (name, (x, y, z)) in enumerate(zip(NAMES, cur)):
            move = f"{np.linalg.norm(cur[j, :2] - old[j, :2]):.3f}" if old is not None else "  -  "
            row = f"{j:2d} {name:<11}{x:5.2f} {y:5.2f} {z:6.2f}  {move}"
            cv2.putText(panel, row, (x0, 56 + j * 14), FONT, 0.34, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.line(panel, (col_w, 24), (col_w, PANEL_H - 4), (70, 70, 70), 1)
    return panel


def main():
    ap = argparse.ArgumentParser(description="Palm Control prototype")
    ap.add_argument("--camera", type=int, default=2, help="camera index (default 2; on this laptop 0 is the IR camera)")
    ap.add_argument("--interval-ms", type=int, default=1000, help="landmark panel refresh, ms (default 1000)")
    ap.add_argument("--reject", type=float, default=0.7, help="max landmark distance for a custom match (lower = stricter)")
    ap.add_argument("--click-drop", type=float, default=0.2, help="finger dip needed for a click (lower = more sensitive)")
    ap.add_argument("--mouse", action="store_true", help="start with real left-click on index-click enabled (toggle with m)")
    ap.add_argument("--width", type=int, default=640, help="requested camera width")
    ap.add_argument("--height", type=int, default=480, help="requested camera height")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    store = GestureStore(reject=args.reject)
    selected = store.names()[0] if store.names() else None
    recording = False
    typing = None  # str while entering a new gesture name
    frame_no = 0
    detectors: dict[str, ClickDetector] = {}
    clicks, flash_until = 0, 0.0
    mouse, mouse_on = Mouse(), args.mouse

    landmarker = vision.GestureRecognizer.create_from_options(
        vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
    )

    shown: dict = {}   # snapshot currently displayed in the panel
    prev: dict = {}    # snapshot before that, for movement
    last_update = 0.0
    t0 = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror: also makes Left/Right labels match your real hands
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = int((time.monotonic() - t0) * 1000)
            result = landmarker.recognize_for_video(mp.Image(mp.ImageFormat.SRGB, rgb), ts)

            draw_hands(frame, result)
            hands = draw_labels(frame, result, store)

            now = time.monotonic()
            for label in list(detectors):
                if label not in hands:
                    detectors.pop(label).reset()
            for label, (arr, _) in hands.items():
                det = detectors.setdefault(label, ClickDetector(drop=args.click_drop))
                if det.update(arr, now):
                    clicks += 1
                    flash_until = now + 0.5
                    if mouse_on:
                        mouse.click(Button.left)
            if detectors:
                draw_click_graph(frame, next(iter(detectors.values())), clicks, now < flash_until, mouse_on)

            frame_no += 1
            if recording and selected and hands and frame_no % 3 == 0:
                label, (arr, _) = next(iter(hands.items()))
                store.add_sample(selected, arr, label)

            now = time.monotonic()
            if (now - last_update) * 1000 >= args.interval_ms:
                prev, shown = shown, snapshot(result)
                last_update = now
            age = int((now - last_update) * 1000)

            panel = render_panel(frame.shape[1], shown, prev, args.interval_ms, age)
            draw_hud(frame, store, selected, recording, typing)
            cv2.imshow("Palm Control", np.vstack([frame, panel]))
            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue
            if typing is not None:  # entering a gesture name
                if key == 13:
                    if typing.strip():
                        selected = typing.strip()
                        store.add_gesture(selected)
                    typing = None
                elif key == 27:
                    typing = None
                elif key in (8, 127):
                    typing = typing[:-1]
                elif 32 <= key < 127:
                    typing += chr(key)
            elif key in (ord("q"), 27):
                break
            elif key == ord("m"):
                mouse_on = not mouse_on
            elif key == ord("n"):
                typing, recording = "", False
            elif key == 9 and store.names():  # Tab
                names = store.names()
                selected = names[(names.index(selected) + 1) % len(names)] if selected in names else names[0]
            elif key == ord(" ") and selected:
                recording = not recording
                if not recording:
                    store.save()
            elif key == ord("x") and selected:
                store.delete(selected)
                recording = False
                selected = store.names()[0] if store.names() else None
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
