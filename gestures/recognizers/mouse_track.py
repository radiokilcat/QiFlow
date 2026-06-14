from __future__ import annotations
from typing import Any
from ..base import GestureRecognizer, GestureEvent

_FIST_TIPS = [8, 12, 16, 20]
_FIST_MCPS = [5,  9, 13, 17]
_SMOOTH_POS = 0.35


def _is_fist(hand: Any) -> bool:
    return sum(1 for t, m in zip(_FIST_TIPS, _FIST_MCPS) if hand[t].y > hand[m].y) >= 3


def _fixed_bbox(
    left: Any, right: Any, size: float
) -> tuple[float, float, float, float]:
    """Square of side `size` (normalized) centered on both wrists' centroid."""
    cx = (left[0].x + right[0].x) / 2
    cy = (left[0].y + right[0].y) / 2
    half = size / 2
    return (
        max(0.0, cx - half),
        max(0.0, cy - half),
        min(1.0, cx + half),
        min(1.0, cy + half),
    )


class MouseTrackRecognizer(GestureRecognizer):
    """
    Left fist activates mouse tracking; right index finger tip controls the cursor.

    When capture_zone_enabled:
      - A fixed-size square is computed from both wrists' centroid the moment
        tracking starts and never moves.
      - The square is always reported in payload so the worker can draw it
        whenever both hands are in frame (not only while tracking).

    After cv2.flip the frame is mirrored:
      user's left hand → MediaPipe "Right", user's right hand → MediaPipe "Left".
    """

    gesture_id = "mouse_track"
    name = "Mouse Track"
    is_multi_hand = True

    def __init__(
        self,
        capture_zone_enabled: bool = True,
        capture_zone_size: float = 0.45,
    ) -> None:
        self._zone_enabled = capture_zone_enabled
        self._zone_size = capture_zone_size
        self._active = False
        self._sx = 0.5
        self._sy = 0.5
        self._bbox: tuple[float, float, float, float] | None = None

    # Called from app/main.py when user saves settings
    def update_settings(self, capture_zone_enabled: bool, capture_zone_size: float) -> None:
        self._zone_enabled = capture_zone_enabled
        self._zone_size = capture_zone_size
        self._bbox = None   # will be recomputed on next activation

    @property
    def current_bbox(self) -> tuple[float, float, float, float] | None:
        return self._bbox

    def process(self, landmarks: Any, frame_time: float) -> GestureEvent | None:
        return None

    def process_all(self, hands: list[Any], frame_time: float) -> list[GestureEvent]:
        left  = next((h for h in hands if h.handedness == "Right"), None)
        right = next((h for h in hands if h.handedness == "Left"),  None)

        if left is None or right is None:
            self._bbox = None
            return self._deactivate(frame_time)

        # Always (re-)compute bbox while two hands are visible
        if self._zone_enabled:
            if not self._active or self._bbox is None:
                self._bbox = _fixed_bbox(left, right, self._zone_size)
            # bbox stays frozen after first capture — never updated
        else:
            self._bbox = None

        if not _is_fist(left):
            return self._deactivate(frame_time)

        # Map pointer (right index tip) into bbox
        tip = right[8]
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
