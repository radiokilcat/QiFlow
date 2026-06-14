from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class Landmark:
    x: float
    y: float
    z: float


@dataclass
class HandLandmarks:
    landmarks: list[Landmark]
    handedness: str = ""  # "Left" or "Right" from the user's perspective

    def __getitem__(self, index: int) -> Landmark:
        return self.landmarks[index]

    def __len__(self) -> int:
        return len(self.landmarks)

    @classmethod
    def from_mediapipe(cls, mp_hand_landmarks: Any, handedness: str = "") -> "HandLandmarks":
        return cls(
            landmarks=[
                Landmark(x=lm.x, y=lm.y, z=lm.z)
                for lm in mp_hand_landmarks
            ],
            handedness=handedness,
        )
