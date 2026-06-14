from __future__ import annotations
import time
from typing import Any
from gestures.base import GestureEvent
from bindings.binding import Binding, TriggerType
from bindings.binding_store import BindingStore
from actions.dispatcher import ActionDispatcher
from actions.registry import ActionRegistry
from confirmation.confirmation_manager import ConfirmationManager
from ui.overlay import IOverlay

_TRIGGER_TO_PHASE = {
    "on_start": "started",
    "on_hold": "updated",
    "on_release": "ended",
}


class BindingEngine:
    def __init__(
        self,
        store: BindingStore,
        dispatcher: ActionDispatcher,
        action_registry: ActionRegistry,
        confirmation_manager: ConfirmationManager,
        overlay: IOverlay,
    ) -> None:
        self._store = store
        self._dispatcher = dispatcher
        self._action_registry = action_registry
        self._confirmation = confirmation_manager
        self._overlay = overlay
        self._cooldowns: dict[str, float] = {}
        self._active_exclusive: set[str] = set()  # gesture_ids of active exclusive bindings

    def process(self, event: GestureEvent) -> None:
        self._confirmation.feed_gesture(event)
        self._confirmation.tick()

        # Track exclusive gesture lifecycle
        if any(b.exclusive and b.gesture_id == event.gesture_id for b in self._store.all()):
            if event.phase in ("started", "updated"):
                self._active_exclusive.add(event.gesture_id)
            elif event.phase == "ended":
                self._active_exclusive.discard(event.gesture_id)

        for binding in self._store.all():
            # While an exclusive gesture is active, suppress all non-exclusive bindings
            if self._active_exclusive and not (
                binding.exclusive and binding.gesture_id in self._active_exclusive
            ):
                continue

            if not self._matches(binding, event):
                continue

            self._validate_high_risk(binding)

            action_preview = self._dispatcher.preview(
                binding.action_id, binding.action_params
            )

            pending = self._confirmation._pending.get(binding.id)
            if pending is not None:
                pending._gesture_id_hint = event.gesture_id
                continue

            def make_confirm_cb(b: Binding, payload: dict) -> Any:
                def cb() -> None:
                    self._dispatcher.execute(b.action_id, b.action_params, event_payload=payload)
                    self._cooldowns[b.id] = time.monotonic() + b.cooldown_ms / 1000.0
                return cb

            confirm = make_confirm_cb(binding, event.payload)
            p = self._confirmation._pending.get(binding.id)
            if p is not None:
                p._gesture_id_hint = event.gesture_id

            self._confirmation.request(
                binding_id=binding.id,
                policy=binding.confirmation,
                action_preview=action_preview,
                on_confirm=confirm,
            )

            if binding.id in self._confirmation._pending:
                self._confirmation._pending[binding.id]._gesture_id_hint = event.gesture_id

    def _matches(self, binding: Binding, event: GestureEvent) -> bool:
        if not binding.enabled:
            return False
        if binding.gesture_id != event.gesture_id:
            return False
        if event.confidence < binding.min_confidence:
            return False
        expected_phase = _TRIGGER_TO_PHASE.get(binding.trigger)
        if expected_phase and event.phase != expected_phase:
            return False
        now = time.monotonic()
        cooldown_until = self._cooldowns.get(binding.id, 0.0)
        if now < cooldown_until:
            return False
        return True

    def _validate_high_risk(self, binding: Binding) -> None:
        try:
            action = self._action_registry.get(binding.action_id)
            if action.risk_level == "high" and binding.confirmation.type == "none":
                raise ValueError(
                    f"Binding '{binding.id}': action '{binding.action_id}' is high-risk "
                    f"and requires confirmation policy != 'none'"
                )
        except KeyError:
            pass
