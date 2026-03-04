from __future__ import annotations

from posture_recognition.gui.posturemirror_app import build_parser


def test_gui_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.backend == "realtime"
    assert args.pose_mode == "upper_body"
    assert args.capture_width == 960
    assert args.capture_height == 540
