from __future__ import annotations
import threading
import time
from queue import Queue, Full

import cv2
import numpy as np

from gestures.registry import GestureRegistry
from gestures.base import GestureEvent
from app.ui.view_models import FramePacket
from config.app_settings import AppSettings

_DISPLAY_W, _DISPLAY_H = 640, 360   # texture size for DearPyGui (fixed)


class CameraWorker(threading.Thread):
    """
    Worker thread: camera capture → MediaPipe detection → GestureRecognizers → queues.

    Detection runs at _DETECT_W×_DETECT_H to keep MediaPipe fast.
    The display frame is pre-converted to float32 RGBA here so the UI thread
    only calls dpg.set_value() without any heavy image work.
    """

    def __init__(
        self,
        settings: AppSettings,
        model_path: str,
        gesture_registry: GestureRegistry,
        frame_queue: Queue[FramePacket],
        gesture_queue: Queue[GestureEvent],
    ) -> None:
        super().__init__(daemon=True, name="CameraWorker")
        self._settings = settings
        self._model_path = model_path
        self._gesture_registry = gesture_registry
        self._frame_queue = frame_queue
        self._gesture_queue = gesture_queue
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def _mouse_track_bbox(self) -> tuple | None:
        try:
            rec = self._gesture_registry.get("mouse_track")
            return rec.current_bbox  # type: ignore[attr-defined]
        except (KeyError, AttributeError):
            return None

    def run(self) -> None:
        from vision.mediapipe_detector import LandmarkDetector
        from vision.drawing import draw_hand_landmarks, draw_tracking_bbox

        try:
            detector = LandmarkDetector(self._model_path)
            detector.start()
        except Exception as exc:
            print(f"[CameraWorker] MediaPipe init failed: {exc}")
            return

        cam_idx = self._settings.camera_index
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print(f"[CameraWorker] Cannot open camera {cam_idx}")
            detector.stop()
            return

        start_time = time.monotonic()

        try:
            while not self._stop_event.is_set():
                t0 = time.monotonic()

                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)
                timestamp_ms = int((time.monotonic() - start_time) * 1000)

                # Detect at configured resolution — read each frame for live updates
                detect_size = self._settings.detect_size
                small = cv2.resize(frame, detect_size)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                try:
                    hand_landmarks_list = detector.detect(rgb_small, timestamp_ms)
                except Exception:
                    hand_landmarks_list = []

                if self._settings.skeleton_only:
                    frame = np.zeros_like(frame)

                if hand_landmarks_list:
                    draw_hand_landmarks(frame, hand_landmarks_list)

                frame_time = time.monotonic()
                latest_gesture_id: str | None = None
                latest_confidence: float = 0.0
                all_events: list[GestureEvent] = []

                for recognizer in self._gesture_registry.all():
                    if recognizer.is_multi_hand:
                        events = recognizer.process_all(hand_landmarks_list, frame_time)
                    else:
                        events = [
                            e for hand_lm in hand_landmarks_list
                            if (e := recognizer.process(hand_lm, frame_time)) is not None
                        ]
                    all_events.extend(events)

                for event in all_events:
                    latest_gesture_id = event.gesture_id
                    latest_confidence = event.confidence
                    try:
                        self._gesture_queue.put_nowait(event)
                    except Full:
                        pass

                # Draw capture zone whenever two hands are present (if enabled)
                if hand_landmarks_list and self._settings.capture_zone_enabled:
                    bbox = self._mouse_track_bbox()
                    if bbox is not None:
                        draw_tracking_bbox(frame, *bbox)

                # Pre-process display frame here, off the main thread
                display = cv2.resize(frame, (_DISPLAY_W, _DISPLAY_H))
                rgba = cv2.cvtColor(display, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
                texture_data = rgba.ravel()

                packet = FramePacket(
                    texture_data=texture_data,
                    timestamp_ms=timestamp_ms,
                    gesture_id=latest_gesture_id,
                    confidence=latest_confidence,
                    has_hands=bool(hand_landmarks_list),
                )

                # Replace stale frame — drain first so maxsize=1 never blocks
                try:
                    self._frame_queue.get_nowait()
                except Exception:
                    pass
                try:
                    self._frame_queue.put_nowait(packet)
                except Full:
                    pass

                # FPS cap — read interval each frame for live updates
                elapsed = time.monotonic() - t0
                remaining = self._settings.frame_interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)

        finally:
            cap.release()
            detector.stop()
