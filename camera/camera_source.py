from __future__ import annotations
from typing import Iterator
import cv2
import numpy as np


class CameraSource:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._index, cv2.CAP_DSHOW)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "CameraSource":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
