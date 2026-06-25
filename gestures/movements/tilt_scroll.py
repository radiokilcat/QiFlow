from __future__ import annotations
from typing import Any, Literal
from ..base import GestureRecognizer, GestureEvent

_SCROLL_INTERVAL = 0.14   # seconds between scroll ticks
_DOWN_THRESHOLD  =  0.25  # dy_norm > this  → pointing down
_UP_THRESHOLD    = -0.15  # dy_norm < this  → pointing up / slightly above horizontal

# After cv2.flip: user's left → MediaPipe "Right"
_MP_HAND = {"left": "Right", "right": "Left"}


class TiltScrollRecognizer(GestureRecognizer):
    """
    Single-hand point pose: tilt index finger down → scroll down,
    tilt slightly above horizontal → scroll up.

    dy_norm = (tip.y − index_MCP.y) / hand_height
    Positive dy_norm = tip below MCP = pointing down.
    """

    gesture_id = "tilt_scroll"
    name = "Tilt Scroll"
    is_multi_hand = False

    def __init__(self, hand: Literal["left", "right"] = "left") -> None:
        self._mp_hand = _MP_HAND[hand]
        self._active_id: str | None = None
        self._last_emit = 0.0

    def process(self, landmarks: Any, frame_time: float) -> GestureEvent | None:
        # Wrong hand — ignore without sending "ended" (that hand can't end our gesture)
        if getattr(landmarks, "handedness", None) != self._mp_hand:
            return None

        # Point pose: index tip above PIP, ≥2 of middle/ring/pinky curled below MCP
        index_extended = landmarks[8].y < landmarks[6].y
        curled = sum([
            landmarks[12].y > landmarks[9].y,
            landmarks[16].y > landmarks[13].y,
            landmarks[20].y > landmarks[17].y,
        ])

        if not (index_extended and curled >= 2):
            return self._end(frame_time)

        # Tilt: index tip (8) vs index MCP (5), normalised by wrist→middle-MCP height
        hand_size = max(abs(landmarks[9].y - landmarks[0].y), 0.05)
        dy_norm = (landmarks[8].y - landmarks[5].y) / hand_size

        if dy_norm > _DOWN_THRESHOLD:
            direction = "tilt_scroll_down"
        elif dy_norm < _UP_THRESHOLD:
            direction = "tilt_scroll_up"
        else:
            return self._end(frame_time)  # neutral zone — in pose but ambiguous

        # New direction or first activation
        if self._active_id != direction:
            self._active_id = direction
            # Pre-subtract interval so the very next frame can emit "updated"
            self._last_emit = frame_time - _SCROLL_INTERVAL
            return GestureEvent(
                gesture_id=direction, confidence=0.90,
                phase="started", timestamp=frame_time,
            )

        # Rate-limited continuous ticks (drives on_hold bindings)
        if frame_time - self._last_emit >= _SCROLL_INTERVAL:
            self._last_emit = frame_time
            return GestureEvent(
                gesture_id=direction, confidence=0.90,
                phase="updated", timestamp=frame_time,
            )

        return None

    def _end(self, frame_time: float) -> GestureEvent | None:
        if self._active_id is not None:
            gid = self._active_id
            self._active_id = None
            return GestureEvent(
                gesture_id=gid, confidence=0.90,
                phase="ended", timestamp=frame_time,
            )
        return None

    def reset(self) -> None:
        self._active_id = None
        self._last_emit = 0.0
