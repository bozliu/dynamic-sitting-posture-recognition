from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import numpy as np


@dataclass
class FramePacket:
    frame: np.ndarray | None
    frame_index: int
    captured_at: float
    done: bool


class LatestFrameBuffer:
    """A thread-safe latest-frame buffer (capacity effectively = 1)."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._frame: np.ndarray | None = None
        self._frame_index = -1
        self._captured_at = 0.0
        self._done = False

    def push(self, frame: np.ndarray, frame_index: int, captured_at: float) -> None:
        with self._lock:
            self._frame = frame
            self._frame_index = frame_index
            self._captured_at = captured_at

    def mark_done(self) -> None:
        with self._lock:
            self._done = True

    def snapshot(self, copy_frame: bool = False) -> FramePacket:
        with self._lock:
            frame = self._frame
            if copy_frame and frame is not None:
                frame = frame.copy()
            return FramePacket(
                frame=frame,
                frame_index=self._frame_index,
                captured_at=self._captured_at,
                done=self._done,
            )
