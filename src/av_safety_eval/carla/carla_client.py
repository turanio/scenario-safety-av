"""Small CARLA client boundary with synchronous-mode lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def require_carla() -> Any:
    """Import the optional CARLA Python API with an actionable error."""

    try:
        import carla
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The CARLA Python API is unavailable. Run this experiment from the "
            "CARLA-enabled av-safety environment."
        ) from exc
    return carla


@dataclass(frozen=True)
class CarlaClientConfig:
    """Connection and deterministic stepping configuration."""

    host: str = "127.0.0.1"
    port: int = 2000
    timeout_seconds: float = 20.0
    town: str | None = None
    fixed_delta_seconds: float = 0.05

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("host must not be empty")
        if self.port <= 0:
            raise ValueError("port must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.fixed_delta_seconds <= 0:
            raise ValueError("fixed_delta_seconds must be positive")


@dataclass
class CollisionRecorder:
    """Event sink for CARLA's synchronous collision callback."""

    available: bool = False
    event_count: int = 0
    note: str = "Collision sensor has not been attached."

    @property
    def collision_detected(self) -> bool:
        return self.event_count > 0

    def __call__(self, _event: Any) -> None:
        self.event_count += 1


class CarlaSession:
    """Own a CARLA connection, actors, and reversible synchronous settings."""

    def __init__(self, config: CarlaClientConfig) -> None:
        self.config = config
        self.carla: Any | None = None
        self.client: Any | None = None
        self.world: Any | None = None
        self._original_settings: Any | None = None
        self._actors: list[Any] = []

    def __enter__(self) -> "CarlaSession":
        carla = require_carla()
        client = carla.Client(self.config.host, self.config.port)
        client.set_timeout(self.config.timeout_seconds)
        world = client.get_world()

        if self.config.town is not None:
            current_map = world.get_map().name.rsplit("/", maxsplit=1)[-1]
            if current_map != self.config.town:
                world = client.load_world(self.config.town)

        self.carla = carla
        self.client = client
        self.world = world
        self._original_settings = world.get_settings()

        try:
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.config.fixed_delta_seconds
            world.apply_settings(settings)
            world.tick()
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

    def tick(self) -> int:
        if self.world is None:
            raise RuntimeError("CARLA session is not connected")
        return int(self.world.tick())

    def spawn_vehicle(
        self,
        transform: Any,
        *,
        role_name: str,
        blueprint_filter: str = "vehicle.tesla.model3",
    ) -> Any:
        if self.world is None:
            raise RuntimeError("CARLA session is not connected")

        library = self.world.get_blueprint_library()
        blueprints = list(library.filter(blueprint_filter))
        if not blueprints:
            blueprints = list(library.filter("vehicle.*"))
        if not blueprints:
            raise RuntimeError("No vehicle blueprints are available in CARLA")

        blueprint = sorted(blueprints, key=lambda item: item.id)[0]
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", role_name)
        actor = self.world.try_spawn_actor(blueprint, transform)
        if actor is None:
            raise RuntimeError(f"Could not spawn CARLA vehicle for role {role_name}")
        self._actors.append(actor)
        return actor

    def attach_collision_sensor(self, vehicle: Any) -> CollisionRecorder:
        recorder = CollisionRecorder()
        if self.world is None or self.carla is None:
            return recorder

        try:
            blueprint = self.world.get_blueprint_library().find(
                "sensor.other.collision"
            )
            sensor = self.world.spawn_actor(
                blueprint,
                self.carla.Transform(),
                attach_to=vehicle,
            )
            sensor.listen(recorder)
            self._actors.append(sensor)
            recorder.available = True
            recorder.note = "Collision status is based on the CARLA collision sensor."
        except (RuntimeError, AttributeError) as exc:
            recorder.note = (
                "Collision sensor unavailable; collision is reported false. "
                f"CARLA message: {exc}"
            )
        return recorder

    def destroy_actors(self) -> None:
        for actor in reversed(self._actors):
            try:
                if hasattr(actor, "stop"):
                    actor.stop()
                if getattr(actor, "is_alive", False):
                    actor.destroy()
            except RuntimeError:
                pass
        self._actors.clear()

    def close(self) -> None:
        self.destroy_actors()
        if self.world is not None and self._original_settings is not None:
            try:
                self.world.apply_settings(self._original_settings)
            except RuntimeError:
                pass
        self.world = None
        self.client = None
        self.carla = None
