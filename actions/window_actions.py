from __future__ import annotations
from typing import Literal, Type
from pydantic import field_validator
from .base import Action, ActionParams, ActionContext, RiskLevel


class WindowSwitchParams(ActionParams):
    direction: Literal["next", "previous"]


class WindowSwitchAction(Action):
    action_id = "window.switch"
    name = "Switch Window"
    description = "Switch to the next or previous window"
    risk_level: RiskLevel = "low"
    params_model: Type[WindowSwitchParams] = WindowSwitchParams

    def preview(self, params: WindowSwitchParams) -> str:
        return f"Switch to {params.direction} window"

    def execute(self, params: WindowSwitchParams, context: ActionContext, event_payload: dict = {}) -> None:
        context.os_adapter.switch_window(params.direction)


class WindowCloseParams(ActionParams):
    pass


class WindowCloseAction(Action):
    action_id = "window.close_active"
    name = "Close Active Window"
    description = "Close the currently active window"
    risk_level: RiskLevel = "high"
    params_model: Type[WindowCloseParams] = WindowCloseParams

    def preview(self, params: WindowCloseParams) -> str:
        return "Close the active window"

    def execute(self, params: WindowCloseParams, context: ActionContext, event_payload: dict = {}) -> None:
        context.os_adapter.close_active_window()


class WindowMinimizeParams(ActionParams):
    pass


class WindowMinimizeAction(Action):
    action_id = "window.minimize_active"
    name = "Minimize Active Window"
    description = "Minimize the currently active window"
    risk_level: RiskLevel = "medium"
    params_model: Type[WindowMinimizeParams] = WindowMinimizeParams

    def preview(self, params: WindowMinimizeParams) -> str:
        return "Minimize the active window"

    def execute(self, params: WindowMinimizeParams, context: ActionContext, event_payload: dict = {}) -> None:
        context.os_adapter.minimize_active_window()


class WindowMaximizeParams(ActionParams):
    pass


class WindowMaximizeAction(Action):
    action_id = "window.maximize_active"
    name = "Maximize Active Window"
    description = "Maximize the currently active window"
    risk_level: RiskLevel = "low"
    params_model: Type[WindowMaximizeParams] = WindowMaximizeParams

    def preview(self, params: WindowMaximizeParams) -> str:
        return "Maximize the active window"

    def execute(self, params: WindowMaximizeParams, context: ActionContext, event_payload: dict = {}) -> None:
        context.os_adapter.maximize_active_window()
