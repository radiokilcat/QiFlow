from __future__ import annotations
from typing import Any
from .landmarks import HandLandmarks


class LandmarkDetector:
    """Thin wrapper around MediaPipe HandLandmarker. Not instantiated in demo mode."""

    def __init__(self, model_path: str, num_hands: int = 2) -> None:
        self._model_path = model_path
        self._landmarker: Any = None
        self._num_hands = num_hands

    def start(self) -> None:
        import mediapipe as mp
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=self._num_hands,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

    def stop(self) -> None:
        if self._landmarker:
            self._landmarker.close()

    def detect(self, rgb_frame: Any, timestamp_ms: int) -> list[HandLandmarks]:
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        hands = []
        for i, hand in enumerate(result.hand_landmarks or []):
            handedness = ""
            if result.handedness and i < len(result.handedness):
                handedness = result.handedness[i][0].category_name
            hands.append(HandLandmarks.from_mediapipe(hand, handedness))
        return hands
