from __future__ import annotations
from typing import Any
from .registry import ActionRegistry
from .base import ActionContext


class ActionDispatcher:
    def __init__(self, registry: ActionRegistry, context: ActionContext) -> None:
        self._registry = registry
        self._context = context

    def preview(self, action_id: str, raw_params: dict[str, Any]) -> str:
        action = self._registry.get(action_id)
        params = action.params_model.model_validate(raw_params)
        return action.preview(params)

    def execute(
        self,
        action_id: str,
        raw_params: dict[str, Any],
        event_payload: dict[str, Any] | None = None,
    ) -> None:
        action = self._registry.get(action_id)
        params = action.params_model.model_validate(raw_params)
        action.execute(params, self._context, event_payload or {})
