from __future__ import annotations
from abc import ABC, abstractmethod


class OSAdapter(ABC):
    @abstractmethod
    def switch_window(self, direction: str) -> None: ...

    @abstractmethod
    def close_active_window(self) -> None: ...

    @abstractmethod
    def minimize_active_window(self) -> None: ...

    @abstractmethod
    def maximize_active_window(self) -> None: ...

    @abstractmethod
    def move_active_window(self, x: int, y: int, width: int | None = None, height: int | None = None) -> None: ...

    @abstractmethod
    def press_hotkey(self, keys: list[str]) -> None: ...

    @abstractmethod
    def move_mouse(self, x: int, y: int) -> None: ...

    @abstractmethod
    def click_mouse(self, button: str) -> None: ...

    @abstractmethod
    def scroll(self, delta: int) -> None: ...

    @abstractmethod
    def change_volume(self, delta: int) -> None: ...
