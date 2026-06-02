from __future__ import annotations
import time
from typing import Callable
from .confirmation_policy import ConfirmationPolicy, ConfirmationType
from gestures.base import GestureEvent
from ui.overlay import IOverlay


PendingCallback = Callable[[], None]


class ConfirmationManager:
    """
    Manages confirmation flow before dispatching actions.
    Does not know about MediaPipe, OSAdapter, or ActionDispatcher directly —
    it receives callbacks and uses the Overlay for user communication.
    """

    def __init__(self, overlay: IOverlay) -> None:
        self._overlay = overlay
        self._pending: dict[str, _PendingConfirmation] = {}

    def request(
        self,
        binding_id: str,
        policy: ConfirmationPolicy,
        action_preview: str,
        on_confirm: PendingCallback,
        on_cancel: PendingCallback | None = None,
    ) -> None:
        if policy.type == "none":
            self._overlay.show_message(f"Executing: {action_preview}")
            on_confirm()
            return

        instruction = self._instruction_text(policy)
        self._overlay.show_confirmation(
            title="Gesture recognized",
            text=action_preview,
            instruction=instruction,
        )
        deadline = time.monotonic() + policy.timeout_ms / 1000.0
        self._pending[binding_id] = _PendingConfirmation(
            policy=policy,
            action_preview=action_preview,
            on_confirm=on_confirm,
            on_cancel=on_cancel,
            deadline=deadline,
            hold_start=time.monotonic() if policy.type == "hold" else None,
        )

    def feed_gesture(self, event: GestureEvent) -> None:
        """Feed an incoming gesture event to pending confirmations."""
        now = time.monotonic()
        expired = [bid for bid, p in self._pending.items() if now > p.deadline]
        for bid in expired:
            pending = self._pending.pop(bid)
            self._overlay.show_message(f"Confirmation timed out: {pending.action_preview}")
            if pending.on_cancel:
                pending.on_cancel()

        for bid, pending in list(self._pending.items()):
            ct = pending.policy.type
            if ct == "hold":
                if event.gesture_id == pending._gesture_id_hint:
                    if pending.hold_start is None:
                        pending.hold_start = now
                    elif (now - pending.hold_start) * 1000 >= pending.policy.hold_ms:
                        self._pending.pop(bid)
                        pending.on_confirm()
                else:
                    pending.hold_start = None
            elif ct == "repeat_gesture":
                if event.gesture_id == pending._gesture_id_hint:
                    self._pending.pop(bid)
                    pending.on_confirm()
            elif ct == "second_gesture":
                if event.gesture_id == pending.policy.second_gesture_id:
                    self._pending.pop(bid)
                    pending.on_confirm()

    def tick(self) -> None:
        """Call periodically to handle hold confirmations via time."""
        now = time.monotonic()
        for bid, pending in list(self._pending.items()):
            if now > pending.deadline:
                self._pending.pop(bid)
                self._overlay.show_message(f"Confirmation timed out: {pending.action_preview}")
                if pending.on_cancel:
                    pending.on_cancel()
                continue
            if pending.policy.type == "hold" and pending.hold_start is not None:
                if (now - pending.hold_start) * 1000 >= pending.policy.hold_ms:
                    self._pending.pop(bid)
                    pending.on_confirm()

    def _instruction_text(self, policy: ConfirmationPolicy) -> str:
        ct = policy.type
        if ct == "hold":
            return f"Hold the gesture for {policy.hold_ms}ms to confirm"
        if ct == "repeat_gesture":
            return "Repeat the gesture to confirm"
        if ct == "second_gesture":
            return f"Show gesture '{policy.second_gesture_id}' to confirm"
        if ct == "hotkey":
            return "Press the confirmation hotkey"
        return "Confirm"


class _PendingConfirmation:
    def __init__(
        self,
        policy: ConfirmationPolicy,
        action_preview: str,
        on_confirm: PendingCallback,
        on_cancel: PendingCallback | None,
        deadline: float,
        hold_start: float | None,
    ) -> None:
        self.policy = policy
        self.action_preview = action_preview
        self.on_confirm = on_confirm
        self.on_cancel = on_cancel
        self.deadline = deadline
        self.hold_start = hold_start
        self._gesture_id_hint: str = ""  # set by BindingEngine after creation
