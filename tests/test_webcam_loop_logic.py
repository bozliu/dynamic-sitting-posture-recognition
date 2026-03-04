from __future__ import annotations

import numpy as np

from posture_recognition.realtime import LatestFrameBuffer


def test_latest_frame_buffer_keeps_latest_only() -> None:
    buffer = LatestFrameBuffer()
    for idx in range(10):
        frame = np.full((4, 4, 3), idx, dtype=np.uint8)
        buffer.push(frame, frame_index=idx, captured_at=float(idx))

    packet = buffer.snapshot(copy_frame=False)
    assert packet.frame_index == 9
    assert packet.frame is not None
    assert int(packet.frame[0, 0, 0]) == 9


def test_latest_frame_buffer_done_flag() -> None:
    buffer = LatestFrameBuffer()
    buffer.push(np.zeros((2, 2, 3), dtype=np.uint8), frame_index=0, captured_at=0.0)
    buffer.mark_done()
    packet = buffer.snapshot(copy_frame=False)
    assert packet.done is True
