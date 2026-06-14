from __future__ import annotations
from .base import OSAdapter


class WindowsAdapter(OSAdapter):
    """Stub implementation. Replace method bodies with pywin32/pyautogui calls."""

    def switch_window(self, direction: str) -> None:
        print(f"[OS] switch_window(direction={direction!r})")

    def close_active_window(self) -> None:
        print("[OS] close_active_window()")

    def minimize_active_window(self) -> None:
        print("[OS] minimize_active_window()")

    def maximize_active_window(self) -> None:
        print("[OS] maximize_active_window()")

    def move_active_window(self, x: int, y: int, width: int | None = None, height: int | None = None) -> None:
        print(f"[OS] move_active_window(x={x}, y={y}, width={width}, height={height})")

    def press_hotkey(self, keys: list[str]) -> None:
        combo = "+".join(keys)
        print(f"[OS] press_hotkey({combo!r})")

    def move_mouse(self, x: int, y: int) -> None:
        import ctypes
        ctypes.windll.user32.SetCursorPos(x, y)

    def click_mouse(self, button: str) -> None:
        print(f"[OS] click_mouse(button={button!r})")

    def scroll(self, delta: int) -> None:
        print(f"[OS] scroll(delta={delta})")

    def change_volume(self, delta: int) -> None:
        print(f"[OS] change_volume(delta={delta})")
