"""Palm Control prototype.

Top half: live webcam with hand landmarks.
Bottom half: landmark values per hand, refreshed every N ms (default 1000),
with the movement (delta) since the previous refresh.

Keys: q / Esc = quit.
"""

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL = Path(__file__).parent / "models" / "hand_landmarker.task"
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
    ap.add_argument("--width", type=int, default=640, help="requested camera width")
    ap.add_argument("--height", type=int, default=480, help="requested camera height")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
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
            result = landmarker.detect_for_video(mp.Image(mp.ImageFormat.SRGB, rgb), ts)

            draw_hands(frame, result)

            now = time.monotonic()
            if (now - last_update) * 1000 >= args.interval_ms:
                prev, shown = shown, snapshot(result)
                last_update = now
            age = int((now - last_update) * 1000)

            panel = render_panel(frame.shape[1], shown, prev, args.interval_ms, age)
            cv2.imshow("Palm Control", np.vstack([frame, panel]))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
