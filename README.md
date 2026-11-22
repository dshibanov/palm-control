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
