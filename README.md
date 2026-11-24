# Palm Control

Webcam hand tracking (MediaPipe) with a live landmark panel.

## Setup

```
uv sync
mkdir -p models
curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
uv run main.py [--camera 2] [--interval-ms 1000]
```

Press `q` or `Esc` to quit. On the dev laptop `/dev/video0` is the IR camera; the color webcam is `/dev/video2`.

## Gestures

Needs `models/gesture_recognizer.task`:
`curl -L -o models/gesture_recognizer.task https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task`

Each hand shows a built-in label (Thumb_Up, Victory, Open_Palm, ...) and a custom label.
Teach custom gestures: `n` + name + Enter, hold the pose, `space` to record (again to stop), `Tab` to switch, `x` to delete.
Samples are saved to `gestures.json`; `--reject` sets match strictness.

## Movement: index click

`motion.py` detects an index-finger tap (finger dips and returns within ~0.8 s) using finger extension
relative to the other fingers, so moving the whole hand doesn't trigger it. A live graph and click counter
are drawn on the video; tune with `--click-drop` (default 0.2, lower = more sensitive).

## Palm RTS (game)

`uv run play.py` — tiny WC3-style game (mine gold, train workers, build houses) driven by hand gestures.
`uv run play.py --mouse` plays with mouse/keyboard (for testing game logic). Tests: `uv run pytest`.

| Action | Gesture |
|---|---|
| Cursor | right-hand index fingertip (changeable in the lab) |
| Select / press buttons | index-finger tap |
| Command (move / mine / help build) | middle-finger tap |
| Box select | pinch thumb+index, move, release |
| Pan camera | left open palm, move hand |
| Build mode / cancel | left Victory (V) / left fist |

Keys: `M` switch hand <-> mouse control, `T` train worker, `B` build house, `Esc` cancel, `WASD`/arrows/screen edges pan, `Q` quit.

## Control lab

One lab, one control at a time: `uv run lab.py -m cursor` or `uv run lab.py -m click` (add `--hand left|right`, default right).
Switch modes live with `Tab` or in the **Params** panel (top-right button, or `P`); `S` saves per hand to `calibration.json`, `R` resets the mode, `play.py` uses the same settings.

| Mode | What you tune |
|---|---|
| `cursor` | `C` calibrate (explanation, then two corners with a 2 s settle before each capture), `T` accuracy test; bottom-right plot of raw vs. filtered cursor x; Params: which hand point the cursor follows, and the smoothing filter (none / ema / oneeuro / kalman) with its params |
| `click` | live graph of the index/middle finger signal, tap counters, `- =` or Params buttons for tap sensitivity (drop); pinch is shown because taps are ignored while pinching |

New controls (map pan, pinch, ...) are a class in `lab_modes.py` registered in `MODES`.
