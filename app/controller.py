from __future__ import annotations
from pathlib import Path
from typing import Any

from actions.registry import ActionRegistry
from actions.dispatcher import ActionDispatcher
from actions.base import ActionContext
from bindings.binding import Binding
from bindings.binding_store import BindingStore
from bindings.binding_engine import BindingEngine
from confirmation.confirmation_manager import ConfirmationManager
from gestures.registry import GestureRegistry
from gestures.base import GestureEvent
from os_control.base import OSAdapter
from ui.overlay import IOverlay


class AppController:
    """
    Bridge between DearPyGuiApp and the core pipeline.
    All methods are called from the main (UI) thread.
    The pipeline itself is stateful but single-threaded — thread safety is
    ensured by calling this only from the main thread.
    """

    def __init__(
        self,
        config_path: Path,
        action_registry: ActionRegistry,
        gesture_registry: GestureRegistry,
        os_adapter: OSAdapter,
        overlay: IOverlay,
    ) -> None:
        self._config_path = config_path
        self._action_registry = action_registry
        self._gesture_registry = gesture_registry
        self._overlay = overlay

        context = ActionContext(os_adapter=os_adapter)
        self._dispatcher = ActionDispatcher(registry=action_registry, context=context)
        self._confirmation = ConfirmationManager(overlay=overlay)
        self._store = BindingStore(config_path)
        self._store.load()

        self._engine = BindingEngine(
            store=self._store,
            dispatcher=self._dispatcher,
            action_registry=action_registry,
            confirmation_manager=self._confirmation,
            overlay=overlay,
        )

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def process_gesture_event(self, event: GestureEvent) -> None:
        self._engine.process(event)

    def tick(self) -> None:
        """Call each frame to advance hold-confirmation timers."""
        self._confirmation.tick()

    # ── Binding management ────────────────────────────────────────────────────

    def get_bindings(self) -> list[Binding]:
        return self._store.all()

    def add_binding(self, raw: dict[str, Any]) -> None:
        binding = Binding.model_validate(raw)
        action = self._action_registry.get(binding.action_id)
        self._store.require_confirmation_for_high_risk(
            action_id=binding.action_id,
            risk_level=action.risk_level,
            confirmation_type=binding.confirmation.type,
        )
        self._store.add(binding)

    def update_binding(self, binding_id: str, updates: dict[str, Any]) -> None:
        binding = self._store.get(binding_id)
        updated = binding.model_copy(update=updates)
        # validate high-risk constraint before committing
        action = self._action_registry.get(updated.action_id)
        self._store.require_confirmation_for_high_risk(
            action_id=updated.action_id,
            risk_level=action.risk_level,
            confirmation_type=updated.confirmation.type,
        )
        self._store.remove(binding_id)
        self._store.add(updated)

    def toggle_binding(self, binding_id: str) -> None:
        binding = self._store.get(binding_id)
        self.update_binding(binding_id, {"enabled": not binding.enabled})

    def remove_binding(self, binding_id: str) -> None:
        self._store.remove(binding_id)

    def save_config(self) -> None:
        self._store.save()

    # ── Registry queries (for UI dropdowns) ───────────────────────────────────

    def get_action_ids(self) -> list[str]:
        return self._action_registry.ids()

    def get_gesture_ids(self) -> list[str]:
        return self._gesture_registry.ids()

    def preview_action(self, action_id: str, params: dict[str, Any]) -> str:
        try:
            return self._dispatcher.preview(action_id, params)
        except Exception as exc:
            return f"[error] {exc}"
