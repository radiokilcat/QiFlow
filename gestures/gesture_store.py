from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Literal, Union, Annotated
from pydantic import BaseModel, Field


# ── Pose configs ───────────────────────────────────────────────────────────────

class FistConfig(BaseModel):
    type: Literal["fist"] = "fist"
    hand: Literal["any", "left", "right"] = "any"
    enabled: bool = True

    def build(self, settings: Any = None) -> Any:
        from gestures.recognizers.fist import FistRecognizer
        return FistRecognizer(hand=self.hand)


class OpenPalmConfig(BaseModel):
    type: Literal["open_palm"] = "open_palm"
    hand: Literal["any", "left", "right"] = "any"
    enabled: bool = True

    def build(self, settings: Any = None) -> Any:
        from gestures.recognizers.open_palm import OpenPalmRecognizer
        return OpenPalmRecognizer(hand=self.hand)


class PinchConfig(BaseModel):
    type: Literal["pinch"] = "pinch"
    hand: Literal["any", "left", "right"] = "any"
    enabled: bool = True

    def build(self, settings: Any = None) -> Any:
        from gestures.recognizers.pinch import PinchRecognizer
        return PinchRecognizer(hand=self.hand)


class PointConfig(BaseModel):
    type: Literal["point"] = "point"
    hand: Literal["any", "left", "right"] = "any"
    enabled: bool = True

    def build(self, settings: Any = None) -> Any:
        from gestures.recognizers.point import PointRecognizer
        return PointRecognizer(hand=self.hand)


# ── Movement configs ───────────────────────────────────────────────────────────

class SwipeLeftConfig(BaseModel):
    type: Literal["swipe_left"] = "swipe_left"

    def build(self, settings: Any = None) -> Any:
        from gestures.movements.swipe import SwipeLeftRecognizer
        return SwipeLeftRecognizer()


class SwipeRightConfig(BaseModel):
    type: Literal["swipe_right"] = "swipe_right"

    def build(self, settings: Any = None) -> Any:
        from gestures.movements.swipe import SwipeRightRecognizer
        return SwipeRightRecognizer()


class MouseTrackConfig(BaseModel):
    type: Literal["mouse_track"] = "mouse_track"
    activator_hand: Literal["left", "right"] = "left"
    activator_pose: Literal["fist", "open_palm", "pinch", "point"] = "fist"
    activation_type: Literal["constant", "toggle"] = "constant"

    def build(self, settings: Any = None) -> Any:
        from gestures.movements.mouse_track import MouseTrackRecognizer
        return MouseTrackRecognizer(
            capture_zone_enabled=settings.capture_zone_enabled if settings else True,
            capture_zone_size=settings.capture_zone_size if settings else 0.45,
            activator_hand=self.activator_hand,
            activator_pose=self.activator_pose,
            activation_type=self.activation_type,
        )


class TiltScrollConfig(BaseModel):
    type: Literal["tilt_scroll"] = "tilt_scroll"
    hand: Literal["left", "right"] = "left"

    def build(self, settings: Any = None) -> Any:
        from gestures.movements.tilt_scroll import TiltScrollRecognizer
        return TiltScrollRecognizer(hand=self.hand)


# ── Discriminated unions ───────────────────────────────────────────────────────

PoseConfig = Annotated[
    Union[FistConfig, OpenPalmConfig, PinchConfig, PointConfig],
    Field(discriminator="type"),
]

MovementConfig = Annotated[
    Union[SwipeLeftConfig, SwipeRightConfig, MouseTrackConfig, TiltScrollConfig],
    Field(discriminator="type"),
]


# ── File model ─────────────────────────────────────────────────────────────────

class GesturesFile(BaseModel):
    poses: list[PoseConfig] = []
    movements: list[MovementConfig] = []


# ── Store ──────────────────────────────────────────────────────────────────────

class GestureStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._data = GesturesFile()

    def load(self) -> None:
        if not self._path.exists():
            self._data = GesturesFile()
            return
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        self._data = GesturesFile.model_validate(raw)

    def save(self) -> None:
        self._path.write_text(
            self._data.model_dump_json(indent=2), encoding="utf-8"
        )

    def build_registry(self, settings: Any = None):
        from gestures.registry import GestureRegistry
        registry = GestureRegistry()
        for cfg in self._data.poses:
            if cfg.enabled:
                registry.register(cfg.build(settings))
        for cfg in self._data.movements:
            registry.register(cfg.build(settings))
        return registry

    @property
    def poses(self) -> list[PoseConfig]:
        return list(self._data.poses)

    @property
    def movements(self) -> list[MovementConfig]:
        return list(self._data.movements)

    def set_poses(self, poses: list[PoseConfig]) -> None:
        self._data = GesturesFile(poses=poses, movements=self._data.movements)

    def set_movements(self, movements: list[MovementConfig]) -> None:
        self._data = GesturesFile(poses=self._data.poses, movements=movements)
