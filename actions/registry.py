from __future__ import annotations
from typing import Iterator
from .base import Action


class ActionRegistry:
    def __init__(self) -> None:
        self._actions: dict[str, Action] = {}

    def register(self, action: Action) -> None:
        if action.action_id in self._actions:
            raise ValueError(f"Action '{action.action_id}' already registered")
        self._actions[action.action_id] = action

    def get(self, action_id: str) -> Action:
        if action_id not in self._actions:
            raise KeyError(f"Action '{action_id}' not found")
        return self._actions[action_id]

    def all(self) -> Iterator[Action]:
        return iter(self._actions.values())

    def ids(self) -> list[str]:
        return list(self._actions.keys())
