"""Timeout-safe key-frame capture for synchronous CARLA experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Condition
import time
from typing import Any


DEFAULT_CAPTURE_STEPS = (0, 40, 70, 100)


@dataclass(frozen=True)
class CameraCaptureConfig:
    """Elevated RGB camera and bounded frame-wait configuration."""

    image_width: int = 960
    image_height: int = 540
    field_of_view_degrees: float = 100.0
    forward_offset_m: float = 8.0
    height_m: float = 30.0
    pitch_degrees: float = -90.0
    image_timeout_seconds: float = 5.0
    capture_steps: tuple[int, ...] = DEFAULT_CAPTURE_STEPS

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("camera image dimensions must be positive")
        if not 1.0 <= self.field_of_view_degrees < 180.0:
            raise ValueError("field_of_view_degrees must be in [1, 180)")
        if self.height_m <= 0:
            raise ValueError("height_m must be positive")
        if self.image_timeout_seconds <= 0:
            raise ValueError("image_timeout_seconds must be positive")
        if any(step < 0 for step in self.capture_steps):
            raise ValueError("capture_steps must be non-negative")
        if len(set(self.capture_steps)) != len(self.capture_steps):
            raise ValueError("capture_steps must be unique")


class LatestRgbFrame:
    """Receive camera callbacks and wait for one exact simulation frame."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._latest_image: Any | None = None

    def __call__(self, image: Any) -> None:
        with self._condition:
            if (
                self._latest_image is None
                or int(image.frame) >= int(self._latest_image.frame)
            ):
                self._latest_image = image
            self._condition.notify_all()

    def save_frame(
        self,
        expected_frame: int,
        output_path: Path,
        timeout_seconds: float,
    ) -> None:
        """Wait at most ``timeout_seconds`` and save the matching CARLA frame."""

        if expected_frame < 0:
            raise ValueError("expected_frame must be non-negative")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while (
                self._latest_image is None
                or int(self._latest_image.frame) < expected_frame
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"RGB camera did not provide frame {expected_frame} within "
                        f"{timeout_seconds:.1f} seconds"
                    )
                self._condition.wait(timeout=remaining)

            image = self._latest_image
            if int(image.frame) != expected_frame:
                raise RuntimeError(
                    f"RGB camera advanced to frame {int(image.frame)} before "
                    f"requested frame {expected_frame} was captured"
                )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save_to_disk(str(output_path))


def key_frame_filename(scenario_name: str, policy_name: str, step: int) -> str:
    """Return the stable thesis-image filename for one rollout step."""

    if not scenario_name or not policy_name:
        raise ValueError("scenario_name and policy_name must not be empty")
    if step < 0:
        raise ValueError("step must be non-negative")
    return f"{scenario_name}_{policy_name}_step_{step:03d}.png"
