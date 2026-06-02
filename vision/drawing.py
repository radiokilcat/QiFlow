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
