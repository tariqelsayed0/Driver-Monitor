# Driver Monitor

Real-time **driver drowsiness and distraction detection** using a webcam.
Built with MediaPipe FaceMesh, OpenCV and Tkinter, with a dark dashboard and an audible alarm.

## Features

- **Eye-closure detection** using the Eye Aspect Ratio (EAR)
- **Yawn detection** using the Mouth Aspect Ratio (MAR)
- **Head pose estimation** (yaw / pitch / roll) via `solvePnP`, reported in real degrees
- **Per-driver calibration** so angles are relative to your natural posture and camera position
- **Time-based rules** (seconds, not frames) so results do not depend on camera FPS
- **Signal smoothing** to avoid false alarms from single noisy frames
- **Face-not-visible alert** when the camera is blocked or the driver leaves
- Alarm cooldown, mute button, and automatic fallback beep if no sound file is found
- Live HUD overlay: face brackets, eye contours, head-direction arrow

## Installation

Requires Python 3.9 - 3.12 and a webcam.

```bash
git clone https://github.com/<your-username>/driver-monitor.git
cd driver_monitor
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python -m driver_monitor
```

Options:

| Flag | Description |
|------|-------------|
| `--camera N` | Camera index (default `0`) |
| `--alarm PATH` | Custom alarm sound (mp3/wav/ogg) |
| `--ear VALUE` | Eye-closure threshold (default `0.22`) |
| `-v` | Debug logging |

On **Start**, the app calibrates for 2 seconds: sit naturally and look straight ahead.
Press **Calibrate** any time you change your seating position.

## How it works

| Signal | Method | Alert when |
|--------|--------|------------|
| Eyes | EAR from 6 eye landmarks | EAR < 0.22 for 1.0 s |
| Yawn | MAR from inner-lip landmarks | MAR > 0.60 for 1.5 s |
| Head | `solvePnP` on 6 face landmarks, Euler angles relative to calibration | yaw > 25 deg, pitch > 20 deg or roll > 25 deg for 1.5 s |
| Presence | FaceMesh detection | no face for 3.0 s |

All thresholds live in [`driver_monitor/config.py`](driver_monitor/config.py).

## Project structure

```
driver-monitor/
├── driver_monitor/
│   ├── __main__.py   # CLI entry point
│   ├── app.py        # Tkinter dashboard
│   ├── monitor.py    # detection engine and state machine
│   ├── metrics.py    # EAR, MAR, head-pose geometry
│   ├── alarm.py      # audio alarm
│   └── config.py     # tunable parameters
├── assets/           # alarm sound
├── requirements.txt
└── LICENSE
```

## Troubleshooting

- **`mp.solutions` not found**: install the pinned version, `pip install mediapipe==0.10.14`.
- **Too many / too few alerts**: adjust thresholds in `config.py` or use `--ear`.
- **Wrong camera**: try `--camera 1`.

## Disclaimer

This is an assistive research and educational project. It is not a certified safety
system and must not replace rest, attention, and responsible driving.

## License

MIT, see [LICENSE](LICENSE).
