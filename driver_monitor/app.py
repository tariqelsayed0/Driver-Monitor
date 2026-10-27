"""Tkinter dashboard."""

from __future__ import annotations

import logging
import sys
import time
import tkinter as tk
from tkinter import messagebox
from typing import Optional

import cv2
from PIL import Image, ImageTk

from .alarm import Alarm
from .config import Config
from .monitor import DriverMonitor, Result, Status

log = logging.getLogger(__name__)

VIDEO_W, VIDEO_H = 640, 480

BG, PANEL, CARD = "#0d1117", "#161b22", "#1c2230"
BORDER, TEXT, MUTED = "#30363d", "#e6edf3", "#8b949e"
GREEN, AMBER, RED, BLUE = "#238636", "#d29922", "#da3633", "#1f6feb"

FONT = "Segoe UI"

STATUS_STYLE = {
    Status.SAFE: ("DRIVER ALERT", GREEN),
    Status.WARNING: ("WARNING", AMBER),
    Status.DANGER: ("DANGER", RED),
    Status.CALIBRATING: ("CALIBRATING", BLUE),
    Status.NO_FACE: ("NO FACE", "#484f58"),
}
HINTS = {
    Status.SAFE: "All good. Keep your eyes on the road.",
    Status.WARNING: "Early signs of fatigue detected.",
    Status.DANGER: "Stay focused!",
    Status.CALIBRATING: "Sit naturally and look straight ahead.",
    Status.NO_FACE: "Looking for a face...",
}


def _shade(hex_color: str, factor: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(min(255, int(c * factor)) for c in (r, g, b))


class FlatButton(tk.Button):
    def __init__(self, parent, text, command, color):
        super().__init__(
            parent, text=text, command=command, relief="flat", bd=0,
            padx=14, pady=10, fg="#ffffff", activeforeground="#ffffff",
            font=(FONT, 10, "bold"), cursor="hand2",
        )
        self.recolor(color)
        self.bind("<Enter>", lambda _e: self.configure(bg=_shade(self._color, 1.18)))
        self.bind("<Leave>", lambda _e: self.configure(bg=self._color))

    def recolor(self, color: str) -> None:
        self._color = color
        self.configure(bg=color, activebackground=_shade(color, 1.3))


class App:
    def __init__(self, root: tk.Tk, cfg: Config) -> None:
        self.root, self.cfg = root, cfg
        self.alarm = Alarm(cfg.alarm_path)
        self.monitor = DriverMonitor(cfg)

        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self._job: Optional[str] = None
        self._read_failures = 0
        self._last_alarm = 0.0
        self._fps = 0.0
        self._t_prev = time.monotonic()

        self._build_ui()
        root.protocol("WM_DELETE_WINDOW", self.close)

    # ------------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        r = self.root
        r.title("Driver Monitor")
        r.configure(bg=BG)
        r.resizable(False, False)

        header = tk.Frame(r, bg=BG)
        header.pack(fill="x", padx=24, pady=(18, 10))
        tk.Label(header, text="DRIVER MONITOR", font=(FONT, 20, "bold"), fg=TEXT, bg=BG).pack(side="left")
        tk.Label(
            header, text="   Drowsiness & distraction detection",
            font=(FONT, 11), fg=MUTED, bg=BG,
        ).pack(side="left", pady=(8, 0))

        body = tk.Frame(r, bg=BG)
        body.pack(padx=24, pady=(0, 10))

        # Video
        frame = tk.Frame(body, bg=BORDER, padx=2, pady=2)
        frame.grid(row=0, column=0, sticky="n")
        box = tk.Frame(frame, bg="#000000", width=VIDEO_W, height=VIDEO_H)
        box.pack()
        box.pack_propagate(False)
        self.video = tk.Label(
            box, text="Camera is off\nPress Start to begin",
            font=(FONT, 13), fg=MUTED, bg="#000000",
        )
        self.video.pack(fill="both", expand=True)

        # Side panel
        side = tk.Frame(body, bg=PANEL, width=320, height=VIDEO_H + 4,
                        highlightbackground=BORDER, highlightthickness=1)
        side.grid(row=0, column=1, padx=(18, 0), sticky="n")
        side.pack_propagate(False)

        self.banner = tk.Label(side, text="READY", font=(FONT, 20, "bold"),
                               fg="#ffffff", bg="#484f58", pady=20)
        self.banner.pack(fill="x", padx=16, pady=(16, 8))

        self.reason = tk.Label(side, text="Press Start to begin monitoring.",
                               font=(FONT, 10), fg=MUTED, bg=PANEL,
                               wraplength=280, height=2)
        self.reason.pack(padx=16)

        grid = tk.Frame(side, bg=PANEL)
        grid.pack(padx=16, pady=10, fill="x")
        self._values = {}
        for i, name in enumerate(("EAR", "MAR", "YAW", "PITCH", "ROLL", "FPS")):
            card = tk.Frame(grid, bg=CARD, padx=12, pady=8)
            card.grid(row=i // 2, column=i % 2, padx=4, pady=4, sticky="nsew")
            grid.columnconfigure(i % 2, weight=1)
            tk.Label(card, text=name, font=(FONT, 8, "bold"), fg=MUTED, bg=CARD).pack(anchor="w")
            value = tk.Label(card, text="--", font=(FONT, 16, "bold"), fg=TEXT, bg=CARD)
            value.pack(anchor="w")
            self._values[name] = value

        buttons = tk.Frame(side, bg=PANEL)
        buttons.pack(side="bottom", fill="x", padx=16, pady=16)
        buttons.columnconfigure((0, 1), weight=1)
        self.btn_start = FlatButton(buttons, "Start", self.toggle, GREEN)
        self.btn_calib = FlatButton(buttons, "Calibrate", self.calibrate, BLUE)
        self.btn_sound = FlatButton(buttons, "Sound: ON", self.toggle_sound, "#30363d")
        self.btn_exit = FlatButton(buttons, "Exit", self.close, RED)
        self.btn_start.grid(row=0, column=0, padx=4, pady=4, sticky="ew")
        self.btn_calib.grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        self.btn_sound.grid(row=1, column=0, padx=4, pady=4, sticky="ew")
        self.btn_exit.grid(row=1, column=1, padx=4, pady=4, sticky="ew")

        tk.Label(
            r, text="Assistive tool only. It does not replace a rested, attentive driver.",
            font=(FONT, 9), fg=MUTED, bg=BG,
        ).pack(pady=(0, 12))

    # -------------------------------------------------------------- actions
    def toggle(self) -> None:
        self.stop() if self.running else self.start()

    def start(self) -> None:
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.cfg.camera_index, backend)
        if not cap.isOpened():
            messagebox.showerror("Camera error", f"Cannot open camera {self.cfg.camera_index}.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)
        self.cap, self.running, self._read_failures = cap, True, 0
        self.monitor.start_calibration()
        self.btn_start.configure(text="Stop")
        self.btn_start.recolor(RED)
        self._loop()

    def stop(self) -> None:
        self.running = False
        if self._job is not None:
            self.root.after_cancel(self._job)
            self._job = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.alarm.stop()
        self.video.configure(image="", text="Camera is off\nPress Start to begin")
        self.video.image = None
        self.banner.configure(text="STOPPED", bg="#484f58")
        self.reason.configure(text="Monitoring stopped.")
        for label in self._values.values():
            label.configure(text="--", fg=TEXT)
        self.btn_start.configure(text="Start")
        self.btn_start.recolor(GREEN)

    def calibrate(self) -> None:
        if self.running:
            self.monitor.start_calibration()

    def toggle_sound(self) -> None:
        self.alarm.muted = not self.alarm.muted
        self.btn_sound.configure(text="Sound: OFF" if self.alarm.muted else "Sound: ON")

    def close(self) -> None:
        self.stop()
        self.monitor.close()
        self.alarm.close()
        self.root.destroy()

    # ----------------------------------------------------------------- loop
    def _loop(self) -> None:
        if not self.running or self.cap is None:
            return
        ok, frame = self.cap.read()
        if ok:
            self._read_failures = 0
            frame, result = self.monitor.process(frame)
            self._update_fps()
            self._render(result)
            self._show(frame)
        else:
            self._read_failures += 1
            if self._read_failures > 30:
                self.stop()
                messagebox.showerror("Camera error", "Lost connection to the camera.")
                return
        self._job = self.root.after(5, self._loop)

    def _update_fps(self) -> None:
        now = time.monotonic()
        dt = now - self._t_prev
        self._t_prev = now
        if dt > 0:
            inst = 1.0 / dt
            self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst

    def _show(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if rgb.shape[1] != VIDEO_W or rgb.shape[0] != VIDEO_H:
            rgb = cv2.resize(rgb, (VIDEO_W, VIDEO_H))
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video.configure(image=photo, text="")
        self.video.image = photo  # keep a reference

    def _render(self, r: Result) -> None:
        c = self.cfg
        text, color = STATUS_STYLE[r.status]
        if r.status is Status.CALIBRATING:
            text = f"CALIBRATING {int(r.calibration_progress * 100)}%"
        self.banner.configure(text=text, bg=color)
        self.reason.configure(text=" | ".join(r.reasons) if r.reasons else HINTS[r.status])

        def put(name: str, value: str, bad: bool) -> None:
            self._values[name].configure(text=value, fg="#ff7b72" if bad else TEXT)

        put("EAR", f"{r.ear:.2f}", r.ear < c.ear_threshold)
        put("MAR", f"{r.mar:.2f}", r.mar > c.mar_threshold)
        put("YAW", f"{r.yaw:+.0f}\u00b0", abs(r.yaw) > c.yaw_limit)
        put("PITCH", f"{r.pitch:+.0f}\u00b0", abs(r.pitch) > c.pitch_limit)
        put("ROLL", f"{r.roll:+.0f}\u00b0", abs(r.roll) > c.roll_limit)
        put("FPS", f"{self._fps:.0f}", False)

        now = time.monotonic()
        if r.status is Status.DANGER and r.fresh and now - self._last_alarm >= c.alarm_cooldown:
            self.alarm.play()
            self._last_alarm = now
