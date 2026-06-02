import time
import pytest
from unittest.mock import MagicMock, patch
from gestures.base import GestureEvent
from bindings.binding import Binding
from bindings.binding_store import BindingStore
from bindings.binding_engine import BindingEngine
from actions.registry import ActionRegistry
from actions.dispatcher import ActionDispatcher
from actions.base import ActionContext
from actions.window_actions import WindowSwitchAction, WindowCloseAction
from confirmation.confirmation_manager import ConfirmationManager
from confirmation.confirmation_policy import ConfirmationPolicy
from os_control.windows_adapter import WindowsAdapter
from ui.overlay import ConsoleOverlay


def _make_store(bindings: list[Binding]) -> BindingStore:
    from pathlib import Path
    store = BindingStore(Path("nonexistent.json"))
    for b in bindings:
        store._bindings.append(b)
    return store


def _make_binding(gesture_id: str = "swipe_left", action_id: str = "window.switch",
                  trigger: str = "on_release", confirmation_type: str = "none",
                  cooldown_ms: int = 0) -> Binding:
    return Binding(
        id=f"binding_{gesture_id}_{action_id}",
        gesture_id=gesture_id,
        action_id=action_id,
        action_params={"direction": "next"} if action_id == "window.switch" else {},
        trigger=trigger,  # type: ignore
        min_confidence=0.7,
        stable_for_ms=0,
        cooldown_ms=cooldown_ms,
        confirmation=ConfirmationPolicy(type=confirmation_type),  # type: ignore
        enabled=True,
    )


def _make_engine(bindings: list[Binding]):
    registry = ActionRegistry()
    registry.register(WindowSwitchAction())
    registry.register(WindowCloseAction())
    adapter = WindowsAdapter()
    context = ActionContext(os_adapter=adapter)
    dispatcher = ActionDispatcher(registry=registry, context=context)
    overlay = ConsoleOverlay()
    cm = ConfirmationManager(overlay=overlay)
    store = _make_store(bindings)
    engine = BindingEngine(
        store=store,
        dispatcher=dispatcher,
        action_registry=registry,
        confirmation_manager=cm,
        overlay=overlay,
    )
    return engine, adapter, cm


def _make_event(gesture_id: str, phase: str, confidence: float = 0.9) -> GestureEvent:
    return GestureEvent(gesture_id=gesture_id, confidence=confidence, phase=phase, timestamp=time.monotonic())  # type: ignore


def test_correct_action_dispatched(capsys):
    binding = _make_binding()
    engine, adapter, _ = _make_engine([binding])
    event = _make_event("swipe_left", "ended")
    engine.process(event)
    captured = capsys.readouterr()
    assert "switch_window" in captured.out


def test_low_confidence_filtered(capsys):
    binding = _make_binding()
    engine, _, _ = _make_engine([binding])
    event = _make_event("swipe_left", "ended", confidence=0.3)
    engine.process(event)
    captured = capsys.readouterr()
    assert "switch_window" not in captured.out


def test_wrong_phase_filtered(capsys):
    binding = _make_binding(trigger="on_release")
    engine, _, _ = _make_engine([binding])
    event = _make_event("swipe_left", "started")
    engine.process(event)
    captured = capsys.readouterr()
    assert "switch_window" not in captured.out


def test_cooldown_prevents_second_dispatch(capsys):
    binding = _make_binding(cooldown_ms=5000)
    engine, _, _ = _make_engine([binding])
    event = _make_event("swipe_left", "ended")
    engine.process(event)
    engine.process(event)
    captured = capsys.readouterr()
    assert captured.out.count("switch_window") == 1


def test_high_risk_without_confirmation_raises():
    binding = _make_binding(
        gesture_id="fist",
        action_id="window.close_active",
        trigger="on_hold",
        confirmation_type="none",
    )
    engine, _, _ = _make_engine([binding])
    event = _make_event("fist", "updated")
    with pytest.raises(ValueError, match="high-risk"):
        engine.process(event)


def test_disabled_binding_skipped(capsys):
    binding = _make_binding()
    binding = binding.model_copy(update={"enabled": False})
    engine, _, _ = _make_engine([binding])
    event = _make_event("swipe_left", "ended")
    engine.process(event)
    captured = capsys.readouterr()
    assert "switch_window" not in captured.out
