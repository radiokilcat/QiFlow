from __future__ import annotations
import sys
import time
from pathlib import Path
from queue import Queue

from actions.registry import ActionRegistry
from actions.dispatcher import ActionDispatcher
from actions.base import ActionContext
from actions.window_actions import (
    WindowSwitchAction, WindowCloseAction,
    WindowMinimizeAction, WindowMaximizeAction,
)
from actions.keyboard_actions import KeyboardHotkeyAction
from actions.mouse_actions import MouseClickAction, MouseScrollAction, MouseMoveAction
from actions.system_actions import VolumeChangeAction

from gestures.registry import GestureRegistry
from gestures.recognizers.open_palm import OpenPalmRecognizer
from gestures.recognizers.fist import FistRecognizer
from gestures.recognizers.pinch import PinchRecognizer
from gestures.recognizers.swipe import SwipeLeftRecognizer, SwipeRightRecognizer
from gestures.recognizers.mouse_track import MouseTrackRecognizer
from gestures.base import GestureEvent

from bindings.binding_store import BindingStore
from bindings.binding_engine import BindingEngine
from confirmation.confirmation_manager import ConfirmationManager
from os_control.windows_adapter import WindowsAdapter
from ui.overlay import ConsoleOverlay

CONFIG_PATH = Path(__file__).parent.parent / "config" / "bindings.json"
MODEL_PATH = Path(__file__).parent.parent / "hand_landmarker.task"


# ── Shared factories ───────────────────────────────────────────────────────────

def build_action_registry() -> ActionRegistry:
    registry = ActionRegistry()
    for action in [
        WindowSwitchAction(),
        WindowCloseAction(),
        WindowMinimizeAction(),
        WindowMaximizeAction(),
        KeyboardHotkeyAction(),
        MouseClickAction(),
        MouseScrollAction(),
        MouseMoveAction(),
        VolumeChangeAction(),
    ]:
        registry.register(action)
    return registry


def build_gesture_registry() -> GestureRegistry:
    registry = GestureRegistry()
    for recognizer in [
        OpenPalmRecognizer(),
        FistRecognizer(),
        PinchRecognizer(),
        SwipeLeftRecognizer(),
        SwipeRightRecognizer(),
        MouseTrackRecognizer(),
    ]:
        registry.register(recognizer)
    return registry


# ── Demo mode (no GUI, no camera) ─────────────────────────────────────────────

def make_event(gesture_id: str, phase: str, confidence: float = 0.9) -> GestureEvent:
    return GestureEvent(
        gesture_id=gesture_id,
        confidence=confidence,
        phase=phase,  # type: ignore[arg-type]
        timestamp=time.monotonic(),
    )


def run_demo() -> None:
    print("=" * 60)
    print("  Gesture Control — Demo Mode")
    print("=" * 60)

    overlay = ConsoleOverlay()
    os_adapter = WindowsAdapter()
    action_registry = build_action_registry()
    gesture_registry = build_gesture_registry()
    context = ActionContext(os_adapter=os_adapter)
    dispatcher = ActionDispatcher(registry=action_registry, context=context)
    confirmation_manager = ConfirmationManager(overlay=overlay)

    store = BindingStore(CONFIG_PATH)
    store.load()
    print(f"\nLoaded {len(store.all())} binding(s) from {CONFIG_PATH}\n")

    engine = BindingEngine(
        store=store,
        dispatcher=dispatcher,
        action_registry=action_registry,
        confirmation_manager=confirmation_manager,
        overlay=overlay,
    )

    demo_events: list[tuple[str, GestureEvent]] = [
        ("Swipe left → switch to next window (no confirmation)",
         make_event("swipe_left", "ended", confidence=0.92)),
        ("Swipe right → switch to previous window (no confirmation)",
         make_event("swipe_right", "ended", confidence=0.85)),
        ("Open palm → minimize window (no confirmation)",
         make_event("open_palm", "started", confidence=0.80)),
        ("Pinch → volume up (no confirmation)",
         make_event("pinch", "started", confidence=0.88)),
        ("Fist hold → close window (requires hold confirmation)",
         make_event("fist", "updated", confidence=0.90)),
        ("Low confidence swipe — should be filtered out",
         make_event("swipe_left", "ended", confidence=0.50)),
    ]

    for description, event in demo_events:
        print(f"\n--- {description} ---")
        print(f"    GestureEvent: gesture_id={event.gesture_id!r}, "
              f"phase={event.phase!r}, confidence={event.confidence:.2f}")
        engine.process(event)

        if event.gesture_id == "fist":
            print("    [simulating hold confirmation — waiting hold_ms...]")
            pending = confirmation_manager._pending.get("binding_close_window")
            if pending:
                pending.hold_start = time.monotonic() - 2.0
            confirmation_manager.tick()

    print("\n" + "=" * 60)
    print("  Demo complete.")
    print("=" * 60)


# ── GUI mode ───────────────────────────────────────────────────────────────────

def run_gui(camera_index: int = 0) -> None:
    from app.ui.view_models import FramePacket, OverlayMessage
    from app.ui.overlay import QueueOverlay
    from app.ui.dearpygui_app import DearPyGuiApp
    from app.controller import AppController
    from app.worker import CameraWorker

    # Queues ──────────────────────────────────────────────────────────────────
    frame_queue: Queue[FramePacket] = Queue(maxsize=1)
    gesture_queue: Queue[GestureEvent] = Queue(maxsize=32)
    overlay_queue: Queue[OverlayMessage] = Queue(maxsize=32)

    # Core objects ────────────────────────────────────────────────────────────
    os_adapter = WindowsAdapter()
    action_registry = build_action_registry()
    gesture_registry = build_gesture_registry()
    queue_overlay = QueueOverlay(overlay_queue)

    controller = AppController(
        config_path=CONFIG_PATH,
        action_registry=action_registry,
        gesture_registry=gesture_registry,
        os_adapter=os_adapter,
        overlay=queue_overlay,
    )

    # Camera worker (daemon — dies when main thread exits) ────────────────────
    if MODEL_PATH.exists():
        worker = CameraWorker(
            camera_index=camera_index,
            model_path=str(MODEL_PATH),
            gesture_registry=gesture_registry,
            frame_queue=frame_queue,
            gesture_queue=gesture_queue,
        )
        worker.start()
    else:
        print(f"[GUI] hand_landmarker.task not found at {MODEL_PATH} — running without camera")
        worker = None

    # DearPyGui app ───────────────────────────────────────────────────────────
    app = DearPyGuiApp(
        frame_queue=frame_queue,
        gesture_queue=gesture_queue,
        overlay_queue=overlay_queue,
        on_gesture_event=controller.process_gesture_event,
        on_controller_tick=controller.tick,
        on_add_binding=controller.add_binding,
        on_save_binding=controller.update_binding,
        on_toggle_binding=controller.toggle_binding,
        on_delete_binding=controller.remove_binding,
        on_save_config=controller.save_config,
        get_bindings=controller.get_bindings,
        get_action_ids=controller.get_action_ids,
        get_gesture_ids=controller.get_gesture_ids,
        preview_action=controller.preview_action,
    )

    app.setup()
    try:
        app.run()
    finally:
        if worker is not None:
            worker.stop()
            worker.join(timeout=2.0)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--gui" in sys.argv:
        camera_idx = 0
        for arg in sys.argv:
            if arg.startswith("--camera="):
                camera_idx = int(arg.split("=", 1)[1])
        run_gui(camera_index=camera_idx)
    else:
        run_demo()
