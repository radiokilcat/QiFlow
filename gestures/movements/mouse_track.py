from __future__ import annotations
import math
from typing import Any, Literal
from ..base import GestureRecognizer, GestureEvent

_SMOOTH_POS = 0.35

# After cv2.flip: user's left → MediaPipe "Right", user's right → MediaPipe "Left"
_MP_ACTIVATOR = {"left": "Right", "right": "Left"}
_MP_POINTER   = {"left": "Left",  "right": "Right"}

_TIPS = [8, 12, 16, 20]
_PIPS = [6, 10, 14, 18]
_MCPS = [5,  9, 13, 17]


def _dist(a: Any, b: Any) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


_POSE_CHECKS: dict[str, Any] = {
    "fist":      lambda h: sum(1 for t, m in zip(_TIPS, _MCPS) if h[t].y > h[m].y) >= 3,
    "open_palm": lambda h: sum(1 for t, p in zip(_TIPS, _PIPS) if h[t].y < h[p].y) >= 3,
    "pinch":     lambda h: (_dist(h[4], h[8]) / (_dist(h[0], h[9]) or 0.001)) < 0.30,
    "point":     lambda h: (h[8].y < h[6].y) and sum([
                     h[12].y > h[9].y, h[16].y > h[13].y, h[20].y > h[17].y
                 ]) >= 2,
}


def _fixed_bbox(
    activator: Any, pointer: Any, size: float
) -> tuple[float, float, float, float]:
    """Square of side `size` (normalized) centered on both wrists' centroid."""
    cx = (activator[0].x + pointer[0].x) / 2
    cy = (activator[0].y + pointer[0].y) / 2
    half = size / 2
    return (
        max(0.0, cx - half),
        max(0.0, cy - half),
        min(1.0, cx + half),
        min(1.0, cy + half),
    )


class MouseTrackRecognizer(GestureRecognizer):
    """
    Activator hand + pose controls mouse tracking; the other hand's index tip
    moves the cursor.

    activation_type="constant": tracking is active while the activator pose is held.
    activation_type="toggle":   activator pose rising-edge toggles tracking on/off;
                                once on, the activator hand can be lowered freely.

    After cv2.flip: user's left → MediaPipe "Right", user's right → MediaPipe "Left".
    """

    gesture_id = "mouse_track"
    name = "Mouse Track"
    is_multi_hand = True

    def __init__(
        self,
        capture_zone_enabled: bool = True,
        capture_zone_size: float = 0.45,
        activator_hand: Literal["left", "right"] = "left",
        activator_pose: Literal["fist", "open_palm", "pinch", "point"] = "fist",
        activation_type: Literal["constant", "toggle"] = "constant",
    ) -> None:
        self._zone_enabled = capture_zone_enabled
        self._zone_size = capture_zone_size
        self._activator_hand = activator_hand
        self._activator_pose = activator_pose
        self._activation_type = activation_type
        self._active = False
        self._sx = 0.5
        self._sy = 0.5
        self._bbox: tuple[float, float, float, float] | None = None
        # toggle mode state
        self._toggled_on = False
        self._pose_was_active = False

    def update_settings(self, capture_zone_enabled: bool, capture_zone_size: float) -> None:
        self._zone_enabled = capture_zone_enabled
        self._zone_size = capture_zone_size
        self._bbox = None

    def update_activator(
        self,
        activator_hand: Literal["left", "right"],
        activator_pose: Literal["fist", "open_palm", "pinch", "point"],
        activation_type: Literal["constant", "toggle"] = "constant",
    ) -> None:
        self._activator_hand = activator_hand
        self._activator_pose = activator_pose
        self._activation_type = activation_type
        self._bbox = None
        self._toggled_on = False
        self._pose_was_active = False

    @property
    def current_bbox(self) -> tuple[float, float, float, float] | None:
        return self._bbox

    def process(self, landmarks: Any, frame_time: float) -> GestureEvent | None:
        return None

    def process_all(self, hands: list[Any], frame_time: float) -> list[GestureEvent]:
        mp_act = _MP_ACTIVATOR[self._activator_hand]
        mp_ptr = _MP_POINTER[self._activator_hand]

        activator = next((h for h in hands if h.handedness == mp_act), None)
        pointer   = next((h for h in hands if h.handedness == mp_ptr), None)

        if self._activation_type == "toggle":
            return self._process_toggle(activator, pointer, frame_time)
        else:
            return self._process_constant(activator, pointer, frame_time)

    # ── Constant mode ──────────────────────────────────────────────────────────

    def _process_constant(
        self, activator: Any, pointer: Any, frame_time: float
    ) -> list[GestureEvent]:
        if activator is None or pointer is None:
            self._bbox = None
            return self._deactivate(frame_time)

        if self._zone_enabled:
            if not self._active or self._bbox is None:
                self._bbox = _fixed_bbox(activator, pointer, self._zone_size)
        else:
            self._bbox = None

        pose_check = _POSE_CHECKS.get(self._activator_pose, _POSE_CHECKS["fist"])
        if not pose_check(activator):
            return self._deactivate(frame_time)

        return self._emit_cursor(pointer, frame_time)

    # ── Toggle mode ────────────────────────────────────────────────────────────

    def _process_toggle(
        self, activator: Any, pointer: Any, frame_time: float
    ) -> list[GestureEvent]:
        pose_check = _POSE_CHECKS.get(self._activator_pose, _POSE_CHECKS["fist"])
        pose_active = activator is not None and pose_check(activator)

        # Rising edge of activator pose → toggle on/off
        if pose_active and not self._pose_was_active:
            self._toggled_on = not self._toggled_on
            if self._toggled_on and self._zone_enabled and activator and pointer:
                self._bbox = _fixed_bbox(activator, pointer, self._zone_size)
            elif not self._toggled_on:
                self._bbox = None

        # Reset edge-detection when activator hand leaves frame
        self._pose_was_active = pose_active if activator is not None else False

        if not self._toggled_on or pointer is None:
            return self._deactivate(frame_time)

        return self._emit_cursor(pointer, frame_time)

    # ── Shared cursor emit ─────────────────────────────────────────────────────

    def _emit_cursor(self, pointer: Any, frame_time: float) -> list[GestureEvent]:
        tip = pointer[8]
        if self._bbox is not None:
            bx0, by0, bx1, by1 = self._bbox
            bw = bx1 - bx0 or 1e-6
            bh = by1 - by0 or 1e-6
            nx = max(0.0, min(1.0, (tip.x - bx0) / bw))
            ny = max(0.0, min(1.0, (tip.y - by0) / bh))
        else:
            nx, ny = tip.x, tip.y

        self._sx = _SMOOTH_POS * nx + (1 - _SMOOTH_POS) * self._sx
        self._sy = _SMOOTH_POS * ny + (1 - _SMOOTH_POS) * self._sy

        phase = "started" if not self._active else "updated"
        self._active = True
        return [GestureEvent(
            gesture_id=self.gesture_id,
            confidence=0.95,
            phase=phase,
            timestamp=frame_time,
            payload={"x": self._sx, "y": self._sy},
        )]

    def _deactivate(self, frame_time: float) -> list[GestureEvent]:
        if self._active:
            self._active = False
            return [GestureEvent(
                gesture_id=self.gesture_id,
                confidence=0.9,
                phase="ended",
                timestamp=frame_time,
            )]
        return []

    def reset(self) -> None:
        self._active = False
        self._bbox = None
        self._toggled_on = False
        self._pose_was_active = False
