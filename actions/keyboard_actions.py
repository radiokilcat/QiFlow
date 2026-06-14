from __future__ import annotations
from typing import Type
from pydantic import field_validator
from .base import Action, ActionParams, ActionContext, RiskLevel

_DANGEROUS_COMBINATIONS = {
    frozenset(["ctrl", "alt", "delete"]),
    frozenset(["alt", "f4"]),
}


class HotkeyParams(ActionParams):
    keys: list[str]

    @field_validator("keys")
    @classmethod
    def keys_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("keys must not be empty")
        return [k.lower() for k in v]


class KeyboardHotkeyAction(Action):
    action_id = "keyboard.hotkey"
    name = "Press Hotkey"
    description = "Press a keyboard shortcut combination"
    risk_level: RiskLevel = "medium"
    params_model: Type[HotkeyParams] = HotkeyParams

    def _is_dangerous(self, keys: list[str]) -> bool:
        return frozenset(keys) in _DANGEROUS_COMBINATIONS

    def preview(self, params: HotkeyParams) -> str:
        combo = " + ".join(params.keys)
        danger = " [DANGEROUS]" if self._is_dangerous(params.keys) else ""
        return f"Press hotkey: {combo}{danger}"

    def execute(self, params: HotkeyParams, context: ActionContext, event_payload: dict = {}) -> None:
        context.os_adapter.press_hotkey(params.keys)
