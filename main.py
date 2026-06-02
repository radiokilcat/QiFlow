import cv2
import mediapipe as mp
import urllib.request
import sys
import time
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def download_model():
    print("Downloading hand landmarker model...", flush=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.", flush=True)


def find_cameras(max_index=8):
    """Return list of available camera indices."""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
        cap.release()
    return available


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def draw_landmarks(frame, hand_landmarks_list):
    h, w = frame.shape[:2]
    for hand_landmarks in hand_landmarks_list:
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0, 200, 80), 2, cv2.LINE_AA)
        for x, y in points:
            cv2.circle(frame, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 4, (0, 150, 60), 1, cv2.LINE_AA)


def draw_hud(frame, cameras, cam_idx):
    h, w = frame.shape[:2]
    # Bottom hint
    cv2.putText(
        frame,
        "Q — quit  |  C — switch camera",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
    )
    # Camera indicator top-left
    label = f"Camera {cameras[cam_idx]}  [{cam_idx + 1}/{len(cameras)}]"
    cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 120), 1, cv2.LINE_AA)


def main():
    if not MODEL_PATH.exists():
        download_model()

    print("Scanning for cameras...", flush=True)
    cameras = find_cameras()
    if not cameras:
        print("Error: no cameras found")
        sys.exit(1)
    print(f"Found cameras: {cameras}", flush=True)

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cam_idx = 0
    cap = open_camera(cameras[cam_idx])
    start_time = time.time()

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Camera lost — try to reopen
                cap.release()
                cap = open_camera(cameras[cam_idx])
                continue

            frame = cv2.flip(frame, 1)

            timestamp_ms = int((time.time() - start_time) * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                draw_landmarks(frame, result.hand_landmarks)

            draw_hud(frame, cameras, cam_idx)
            cv2.imshow("Gestures Control", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                cam_idx = (cam_idx + 1) % len(cameras)
                cap.release()
                cap = open_camera(cameras[cam_idx])
                start_time = time.time()
                print(f"Switched to camera {cameras[cam_idx]}", flush=True)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
