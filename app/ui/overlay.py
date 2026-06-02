from __future__ import annotations
from queue import Queue, Full

from ui.overlay import IOverlay
from .view_models import OverlayMessage


class QueueOverlay(IOverlay):
    """
    Thread-safe IOverlay implementation.
    The worker / pipeline threads call show_* methods which enqueue messages.
    The main (UI) thread drains the queue each frame and updates the display.
    """

    def __init__(self, queue: Queue[OverlayMessage]) -> None:
        self._queue = queue

    def show_message(self, text: str) -> None:
        self._put(OverlayMessage(kind="message", text=text))

    def show_confirmation(self, title: str, text: str, instruction: str) -> None:
        self._put(OverlayMessage(kind="confirmation", text=text, title=title, instruction=instruction))

    def show_error(self, text: str) -> None:
        self._put(OverlayMessage(kind="error", text=text))

    def _put(self, msg: OverlayMessage) -> None:
        try:
            self._queue.put_nowait(msg)
        except Full:
            pass
