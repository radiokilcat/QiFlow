from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Type
from pydantic import BaseModel
from os_control.base import OSAdapter


RiskLevel = Literal["low", "medium", "high"]


@dataclass
class ActionContext:
    os_adapter: OSAdapter


class ActionParams(BaseModel):
    """Base class for all action parameter models."""
    pass


class Action(ABC):
    action_id: str
    name: str
    description: str
    risk_level: RiskLevel
    params_model: Type[ActionParams]

    @abstractmethod
    def preview(self, params: ActionParams) -> str:
        """Return a human-readable preview of what this action will do."""
        ...

    @abstractmethod
    def execute(self, params: ActionParams, context: ActionContext) -> None:
        """Execute the action using the OS adapter from context."""
        ...
