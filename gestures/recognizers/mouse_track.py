from __future__ import annotations
from typing import Any
from ..base import GestureRecognizer, GestureEvent

_FIST_TIPS = [8, 12, 16, 20]
_FIST_MCPS = [5,  9, 13, 17]

_SMOOTH_POS = 0.35   # pointer position smoothing
_PADDING    = 0.15   # fraction of box size added on each side


def _is_fist(hand: Any) -> bool:
    return sum(1 for t, m in zip(_FIST_TIPS, _FIST_MCPS) if hand[t].y > hand[m].y) >= 3


def _raw_bbox(hands: list[Any]) -> tuple[float, float, float, float]:
    xs = [lm.x for h in hands for lm in h.landmarks]
    ys = [lm.y for h in hands for lm in h.landmarks]
    return min(xs), min(ys), max(xs), max(ys)


def _padded(x0: float, y0: float, x1: float, y1: float, pad: float) -> tuple[float, float, float, float]:
    pw = (x1 - x0) * pad
    ph = (y1 - y0) * pad
    return (
        max(0.0, x0 - pw),
        max(0.0, y0 - ph),
        min(1.0, x1 + pw),
        min(1.0, y1 + ph),
    )


class MouseTrackRecognizer(GestureRecognizer):
    """
    Left fist activates mouse tracking; right index finger tip controls the cursor.
    Mouse coords are mapped relative to the bounding box of both hands, not the full frame.

    After cv2.flip the frame is mirrored, so MediaPipe swaps handedness labels:
    user's left hand → "Right", user's right hand → "Left".
    """

    gesture_id = "mouse_track"
    name = "Mouse Track"
    is_multi_hand = True

    def __init__(self, padding: float = _PADDING) -> None:
        self._padding = padding
        self._active = False
        self._sx = 0.5
        self._sy = 0.5
        self._bx0 = 0.0
        self._by0 = 0.0
        self._bx1 = 1.0
        self._by1 = 1.0

    def process(self, landmarks: Any, frame_time: float) -> GestureEvent | None:
        return None

    def process_all(self, hands: list[Any], frame_time: float) -> list[GestureEvent]:
        left  = next((h for h in hands if h.handedness == "Right"), None)  # mirrored label
        right = next((h for h in hands if h.handedness == "Left"),  None)

        if left is None or right is None or not _is_fist(left):
            return self._deactivate(frame_time)

        # Bbox is captured once on activation and never updated
        if not self._active:
            self._bx0, self._by0, self._bx1, self._by1 = _padded(*_raw_bbox([left, right]), self._padding)

        bw = self._bx1 - self._bx0 or 1e-6
        bh = self._by1 - self._by0 or 1e-6

        tip = right[8]
        nx = max(0.0, min(1.0, (tip.x - self._bx0) / bw))
        ny = max(0.0, min(1.0, (tip.y - self._by0) / bh))

        self._sx = _SMOOTH_POS * nx + (1 - _SMOOTH_POS) * self._sx
        self._sy = _SMOOTH_POS * ny + (1 - _SMOOTH_POS) * self._sy

        phase = "started" if not self._active else "updated"
        self._active = True
        return [GestureEvent(
            gesture_id=self.gesture_id,
            confidence=0.95,
            phase=phase,
            timestamp=frame_time,
            payload={
                "x":    self._sx,
                "y":    self._sy,
                "bbox": (self._bx0, self._by0, self._bx1, self._by1),
            },
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
