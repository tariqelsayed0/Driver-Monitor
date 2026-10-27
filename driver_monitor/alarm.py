"""Audio alarm with a synthesized fallback when no sound file is available."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

log = logging.getLogger(__name__)

SAMPLE_RATE = 44100


class Alarm:
    def __init__(self, path: Path) -> None:
        self.muted = False
        self._sound: Optional[pygame.mixer.Sound] = None
        try:
            pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
            self._sound = self._load(path)
        except pygame.error as exc:
            log.warning("Audio unavailable, alarm will be silent: %s", exc)

    def _load(self, path: Path) -> pygame.mixer.Sound:
        if path.is_file():
            try:
                return pygame.mixer.Sound(str(path))
            except pygame.error as exc:
                log.warning("Could not load %s (%s); using built-in beep.", path, exc)
        else:
            log.info("%s not found; using built-in beep.", path)
        return self._beep()

    @staticmethod
    def _beep(freq: float = 880.0) -> pygame.mixer.Sound:
        t = np.linspace(0, 0.25, int(SAMPLE_RATE * 0.25), endpoint=False)
        tone = 0.5 * 32767 * np.sin(2 * np.pi * freq * t)
        gap = np.zeros(int(SAMPLE_RATE * 0.1))
        wave = np.concatenate([tone, gap, tone, gap, tone]).astype(np.int16)
        return pygame.sndarray.make_sound(np.ascontiguousarray(np.column_stack((wave, wave))))

    def play(self) -> None:
        if self._sound is None or self.muted or pygame.mixer.get_busy():
            return
        self._sound.play()

    def stop(self) -> None:
        if self._sound is not None:
            self._sound.stop()

    def close(self) -> None:
        if pygame.mixer.get_init():
            pygame.mixer.quit()
