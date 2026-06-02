from __future__ import annotations
from abc import ABC, abstractmethod


class IOverlay(ABC):
    @abstractmethod
    def show_message(self, text: str) -> None: ...

    @abstractmethod
    def show_confirmation(self, title: str, text: str, instruction: str) -> None: ...

    @abstractmethod
    def show_error(self, text: str) -> None: ...


class ConsoleOverlay(IOverlay):
    """Simple console-based overlay for development and demo mode."""

    def show_message(self, text: str) -> None:
        print(f"[OVERLAY] {text}")

    def show_confirmation(self, title: str, text: str, instruction: str) -> None:
        print(f"[CONFIRM] {title}")
        print(f"          Action: {text}")
        print(f"          To confirm: {instruction}")

    def show_error(self, text: str) -> None:
        print(f"[ERROR] {text}")
