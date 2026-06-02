import pytest
from pydantic import ValidationError
from actions.registry import ActionRegistry
from actions.window_actions import WindowSwitchAction, WindowSwitchParams, WindowCloseAction
from actions.keyboard_actions import KeyboardHotkeyAction, HotkeyParams
from actions.base import ActionContext
from os_control.windows_adapter import WindowsAdapter


@pytest.fixture
def registry() -> ActionRegistry:
    r = ActionRegistry()
    r.register(WindowSwitchAction())
    r.register(WindowCloseAction())
    r.register(KeyboardHotkeyAction())
    return r


@pytest.fixture
def context() -> ActionContext:
    return ActionContext(os_adapter=WindowsAdapter())


def test_register_action(registry: ActionRegistry) -> None:
    assert "window.switch" in registry.ids()
    assert "window.close_active" in registry.ids()


def test_duplicate_registration_raises(registry: ActionRegistry) -> None:
    with pytest.raises(ValueError):
        registry.register(WindowSwitchAction())


def test_get_unknown_action_raises(registry: ActionRegistry) -> None:
    with pytest.raises(KeyError):
        registry.get("nonexistent.action")


def test_window_switch_params_valid() -> None:
    p = WindowSwitchParams(direction="next")
    assert p.direction == "next"


def test_window_switch_params_invalid() -> None:
    with pytest.raises(ValidationError):
        WindowSwitchParams(direction="sideways")


def test_hotkey_params_lowercased() -> None:
    p = HotkeyParams(keys=["CTRL", "C"])
    assert p.keys == ["ctrl", "c"]


def test_hotkey_params_empty_raises() -> None:
    with pytest.raises(ValidationError):
        HotkeyParams(keys=[])


def test_action_preview(context: ActionContext) -> None:
    action = WindowSwitchAction()
    params = WindowSwitchParams(direction="previous")
    assert "previous" in action.preview(params)


def test_high_risk_level() -> None:
    action = WindowCloseAction()
    assert action.risk_level == "high"
