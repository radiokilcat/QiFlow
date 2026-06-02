import json
import pytest
from pathlib import Path
from pydantic import ValidationError
from bindings.binding import Binding
from bindings.binding_store import BindingStore
from confirmation.confirmation_policy import ConfirmationPolicy


def make_binding(**kwargs) -> Binding:
    defaults = dict(
        id="test_binding",
        gesture_id="swipe_left",
        action_id="window.switch",
        action_params={"direction": "next"},
        trigger="on_release",
        min_confidence=0.8,
        stable_for_ms=200,
        cooldown_ms=800,
        confirmation=ConfirmationPolicy(type="none"),
        enabled=True,
    )
    defaults.update(kwargs)
    return Binding(**defaults)


def test_binding_valid() -> None:
    b = make_binding()
    assert b.gesture_id == "swipe_left"


def test_binding_confidence_out_of_range() -> None:
    with pytest.raises(ValidationError):
        make_binding(min_confidence=1.5)


def test_binding_store_load(tmp_path: Path) -> None:
    config = {
        "bindings": [
            {
                "id": "b1",
                "gesture_id": "swipe_left",
                "action_id": "window.switch",
                "action_params": {"direction": "next"},
                "trigger": "on_release",
                "min_confidence": 0.8,
                "stable_for_ms": 200,
                "cooldown_ms": 800,
                "confirmation": {"type": "none", "timeout_ms": 1500, "hold_ms": 700, "second_gesture_id": None},
                "enabled": True,
            }
        ]
    }
    p = tmp_path / "bindings.json"
    p.write_text(json.dumps(config))
    store = BindingStore(p)
    store.load()
    assert len(store.all()) == 1
    assert store.all()[0].id == "b1"


def test_binding_store_add_duplicate_raises(tmp_path: Path) -> None:
    store = BindingStore(tmp_path / "bindings.json")
    b = make_binding()
    store.add(b)
    with pytest.raises(ValueError):
        store.add(b)


def test_high_risk_without_confirmation_raises() -> None:
    store = BindingStore(Path("nonexistent.json"))
    with pytest.raises(ValueError):
        store.require_confirmation_for_high_risk(
            action_id="window.close_active",
            risk_level="high",
            confirmation_type="none",
        )


def test_high_risk_with_confirmation_ok() -> None:
    store = BindingStore(Path("nonexistent.json"))
    # Should not raise
    store.require_confirmation_for_high_risk(
        action_id="window.close_active",
        risk_level="high",
        confirmation_type="hold",
    )
