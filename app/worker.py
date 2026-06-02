from __future__ import annotations
import threading
import time
from queue import Queue, Full

from gestures.registry import GestureRegistry
from gestures.base import GestureEvent
from app.ui.view_models import FramePacket


class CameraWorker(threading.Thread):
    """
    Worker thread: camera capture → MediaPipe detection → GestureRecognizers → queues.

    Runs on a daemon thread separate from the main (UI) thread.
    Puts only the latest frame into frame_queue (maxsize=1, old frames replaced).
    Puts GestureEvents into gesture_queue (maxsize=32, drops if full).
    """

    def __init__(
        self,
        camera_index: int,
        model_path: str,
        gesture_registry: GestureRegistry,
        frame_queue: Queue[FramePacket],
        gesture_queue: Queue[GestureEvent],
    ) -> None:
        super().__init__(daemon=True, name="CameraWorker")
        self._camera_index = camera_index
        self._model_path = model_path
        self._gesture_registry = gesture_registry
        self._frame_queue = frame_queue
        self._gesture_queue = gesture_queue
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        import cv2
        from vision.mediapipe_detector import LandmarkDetector
        from vision.drawing import draw_hand_landmarks

        try:
            detector = LandmarkDetector(self._model_path)
            detector.start()
        except Exception as exc:
            print(f"[CameraWorker] MediaPipe init failed: {exc}")
            return

        cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print(f"[CameraWorker] Cannot open camera {self._camera_index}")
            detector.stop()
            return

        start_time = time.monotonic()

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)
                timestamp_ms = int((time.monotonic() - start_time) * 1000)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    hand_landmarks_list = detector.detect(rgb, timestamp_ms)
                except Exception:
                    hand_landmarks_list = []

                if hand_landmarks_list:
                    draw_hand_landmarks(frame, hand_landmarks_list)

                frame_time = time.monotonic()
                latest_gesture_id: str | None = None
                latest_confidence: float = 0.0

                for recognizer in self._gesture_registry.all():
                    for hand_lm in hand_landmarks_list:
                        event = recognizer.process(hand_lm, frame_time)
                        if event is not None:
                            latest_gesture_id = event.gesture_id
                            latest_confidence = event.confidence
                            try:
                                self._gesture_queue.put_nowait(event)
                            except Full:
                                pass

                packet = FramePacket(
                    frame=frame,
                    timestamp_ms=timestamp_ms,
                    gesture_id=latest_gesture_id,
                    confidence=latest_confidence,
                    has_hands=bool(hand_landmarks_list),
                )

                # Replace stale frame: drain first so maxsize=1 never blocks
                try:
                    self._frame_queue.get_nowait()
                except Exception:
                    pass
                try:
                    self._frame_queue.put_nowait(packet)
                except Full:
                    pass

        finally:
            cap.release()
            detector.stop()
