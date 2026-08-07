from pathlib import Path

import pytest

from av_safety_eval.carla.image_capture import (
    CameraCaptureConfig,
    LatestRgbFrame,
    key_frame_filename,
)


class FakeCarlaImage:
    def __init__(self, frame: int) -> None:
        self.frame = frame

    def save_to_disk(self, output_path: str) -> None:
        Path(output_path).write_bytes(b"fake-png")


def test_latest_rgb_frame_saves_exact_matching_frame(tmp_path) -> None:
    receiver = LatestRgbFrame()
    receiver(FakeCarlaImage(frame=42))
    output_path = tmp_path / "frame.png"

    receiver.save_frame(42, output_path, timeout_seconds=0.1)

    assert output_path.read_bytes() == b"fake-png"


def test_latest_rgb_frame_times_out_without_camera_data(tmp_path) -> None:
    receiver = LatestRgbFrame()

    with pytest.raises(TimeoutError, match="did not provide frame 12"):
        receiver.save_frame(12, tmp_path / "missing.png", timeout_seconds=0.01)


def test_latest_rgb_frame_rejects_a_skipped_frame(tmp_path) -> None:
    receiver = LatestRgbFrame()
    receiver(FakeCarlaImage(frame=43))

    with pytest.raises(RuntimeError, match="advanced to frame 43"):
        receiver.save_frame(42, tmp_path / "frame.png", timeout_seconds=0.1)


def test_camera_capture_defaults_and_filename() -> None:
    config = CameraCaptureConfig()

    assert config.capture_steps == (0, 40, 70, 100)
    assert config.image_timeout_seconds == pytest.approx(5.0)
    assert (
        key_frame_filename("near_miss_style", "top1_policy", 40)
        == "near_miss_style_top1_policy_step_040.png"
    )
