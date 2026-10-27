"""Entry point: `python -m driver_monitor`."""

from __future__ import annotations

import argparse
import logging
import tkinter as tk
from dataclasses import replace
from pathlib import Path

from . import __version__
from .app import App
from .config import Config


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="driver-monitor",
        description="Real-time driver drowsiness and distraction detection.",
    )
    parser.add_argument("--camera", type=int, help="camera index (default: 0)")
    parser.add_argument("--alarm", type=Path, help="path to an alarm sound file")
    parser.add_argument("--ear", type=float, help="eye-closure threshold (default: 0.22)")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    overrides = {}
    if args.camera is not None:
        overrides["camera_index"] = args.camera
    if args.alarm is not None:
        overrides["alarm_path"] = args.alarm
    if args.ear is not None:
        overrides["ear_threshold"] = args.ear
    cfg = replace(Config(), **overrides)

    root = tk.Tk()
    App(root, cfg)
    root.mainloop()


if __name__ == "__main__":
    main()
