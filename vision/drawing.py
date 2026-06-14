from __future__ import annotations
import cv2
import numpy as np
from .landmarks import HandLandmarks

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_tracking_bbox(
    frame: np.ndarray,
    x0: float, y0: float, x1: float, y1: float,
) -> None:
    """Draw a tracking zone rectangle with corner accents."""
    h, w = frame.shape[:2]
    p0 = (int(x0 * w), int(y0 * h))
    p1 = (int(x1 * w), int(y1 * h))
    cl = (int((x1 + x0) / 2 * w), int((y1 + y0) / 2 * h))

    # Dim rectangle fill
    overlay = frame.copy()
    cv2.rectangle(overlay, p0, p1, (0, 200, 200), -1)
    cv2.addWeighted(overlay, 0.07, frame, 0.93, 0, frame)

    # Border
    cv2.rectangle(frame, p0, p1, (0, 220, 220), 1, cv2.LINE_AA)

    # Corner accents
    corner = max(12, min(p1[0] - p0[0], p1[1] - p0[1]) // 6)
    color, thick = (0, 240, 240), 2
    for cx, cy, sx, sy in [
        (p0[0], p0[1],  1,  1),
        (p1[0], p0[1], -1,  1),
        (p0[0], p1[1],  1, -1),
        (p1[0], p1[1], -1, -1),
    ]:
        cv2.line(frame, (cx, cy), (cx + sx * corner, cy), color, thick, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + sy * corner), color, thick, cv2.LINE_AA)

    cv2.putText(frame, "TRACK", (p0[0] + 4, p0[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 240, 240), 1, cv2.LINE_AA)


def draw_hand_landmarks(frame: np.ndarray, hand_landmarks_list: list[HandLandmarks]) -> None:
    """Draw skeleton overlay on frame in-place."""
    h, w = frame.shape[:2]
    for hand in hand_landmarks_list:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand.landmarks]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, pts[start], pts[end], (0, 200, 80), 2, cv2.LINE_AA)
        for x, y in pts:
            cv2.circle(frame, (x, y), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 5, (0, 150, 60),  1,  cv2.LINE_AA)
