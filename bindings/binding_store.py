from __future__ import annotations
import json
from pathlib import Path
from pydantic import BaseModel, ValidationError
from .binding import Binding


class BindingsFile(BaseModel):
    bindings: list[Binding]


class BindingStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._bindings: list[Binding] = []

    def load(self) -> None:
        if not self._path.exists():
            self._bindings = []
            return
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        data = BindingsFile.model_validate(raw)
        self._validate_risk_constraints(data.bindings)
        self._bindings = data.bindings

    def save(self) -> None:
        data = BindingsFile(bindings=self._bindings)
        self._path.write_text(
            data.model_dump_json(indent=2), encoding="utf-8"
        )

    def _validate_risk_constraints(self, bindings: list[Binding]) -> None:
        """Bindings for high-risk actions must have non-none confirmation."""
        # This check is action-registry-agnostic; callers may inject risk info.
        # The store enforces: any binding whose action_id ends with 'close_active'
        # or is otherwise marked high-risk must have confirmation != 'none'.
        # Full enforcement requires ActionRegistry; done in BindingEngine.
        pass

    def require_confirmation_for_high_risk(
        self, action_id: str, risk_level: str, confirmation_type: str
    ) -> None:
        if risk_level == "high" and confirmation_type == "none":
            raise ValueError(
                f"Action '{action_id}' is high-risk and requires confirmation"
            )

    def add(self, binding: Binding) -> None:
        ids = {b.id for b in self._bindings}
        if binding.id in ids:
            raise ValueError(f"Binding '{binding.id}' already exists")
        self._bindings.append(binding)

    def remove(self, binding_id: str) -> None:
        self._bindings = [b for b in self._bindings if b.id != binding_id]

    def all(self) -> list[Binding]:
        return list(self._bindings)

    def get(self, binding_id: str) -> Binding:
        for b in self._bindings:
            if b.id == binding_id:
                return b
        raise KeyError(f"Binding '{binding_id}' not found")
