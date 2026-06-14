from __future__ import annotations
import time
from queue import Queue, Empty
from typing import Callable, Any

import numpy as np
import dearpygui.dearpygui as dpg

from gestures.base import GestureEvent
from .view_models import FramePacket, OverlayMessage, ActionLogEntry

# ── Layout constants ───────────────────────────────────────────────────────────
_CAMERA_W = 640
_CAMERA_H = 360
_TEXTURE_TAG = "camera_texture"
_LOG_MAX = 30
_OVERLAY_MAX = 6

# ── Colour palette ─────────────────────────────────────────────────────────────
_COL_HEADER = (220, 200, 100, 255)
_COL_GESTURE = (100, 220, 100, 255)
_COL_GESTURE_ID = (100, 200, 255, 255)
_COL_ACTION_ID = (255, 200, 100, 255)
_COL_MUTED = (160, 160, 160, 255)
_COL_ON = (100, 220, 100, 255)
_COL_OFF = (200, 80, 80, 255)
_COL_ERROR = (220, 80, 80, 255)
_COL_CONFIRM = (220, 180, 60, 255)


class DearPyGuiApp:
    """
    Main-thread UI.  DearPyGui is single-threaded by design — every dpg.*
    call happens here, either inside the render loop (tick) or in UI callbacks
    which DPG calls synchronously on the same thread.

    External state comes in via thread-safe queues:
      frame_queue   – latest camera frame (maxsize=1)
      gesture_queue – GestureEvents from the recogniser (maxsize=32)
      overlay_queue – messages from ConfirmationManager / pipeline (maxsize=32)

    Commands leave via callbacks passed in at construction time.
    """

    def __init__(
        self,
        frame_queue: Queue[FramePacket],
        gesture_queue: Queue[GestureEvent],
        overlay_queue: Queue[OverlayMessage],
        on_gesture_event: Callable[[GestureEvent], None],
        on_controller_tick: Callable[[], None],
        on_save_binding: Callable[[str, dict[str, Any]], None],
        on_toggle_binding: Callable[[str], None],
        on_delete_binding: Callable[[str], None],
        on_save_config: Callable[[], None],
        get_bindings: Callable[[], list],
        get_action_ids: Callable[[], list[str]],
        get_gesture_ids: Callable[[], list[str]],
        preview_action: Callable[[str, dict[str, Any]], str],
    ) -> None:
        self._frame_q = frame_queue
        self._gesture_q = gesture_queue
        self._overlay_q = overlay_queue

        self._on_gesture_event = on_gesture_event
        self._on_controller_tick = on_controller_tick
        self._on_save_binding = on_save_binding
        self._on_toggle_binding = on_toggle_binding
        self._on_delete_binding = on_delete_binding
        self._on_save_config = on_save_config
        self._get_bindings = get_bindings
        self._get_action_ids = get_action_ids
        self._get_gesture_ids = get_gesture_ids
        self._preview_action = preview_action

        self._overlay_msgs: list[OverlayMessage] = []
        self._action_log: list[ActionLogEntry] = []
        self._edit_binding_id: str | None = None
        self._bindings_dirty = True   # rebuild table on first tick

    # ── Setup ──────────────────────────────────────────────────────────────────

    def setup(self) -> None:
        dpg.create_context()
        dpg.create_viewport(
            title="Gesture Control",
            width=1200,
            height=720,
            resizable=True,
            min_width=900,
            min_height=600,
        )
        self._create_texture()
        self._build_ui()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_viewport_resize_callback(self._on_viewport_resize)

    def _create_texture(self) -> None:
        blank = np.zeros((_CAMERA_H, _CAMERA_W, 4), dtype=np.float32)
        with dpg.texture_registry():
            dpg.add_raw_texture(
                width=_CAMERA_W,
                height=_CAMERA_H,
                default_value=blank.ravel(),
                format=dpg.mvFormat_Float_rgba,
                tag=_TEXTURE_TAG,
            )

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        with dpg.window(
            tag="main_window",
            no_title_bar=True,
            no_move=True,
            no_resize=True,
            no_scrollbar=True,
            menubar=True,
        ):
            self._build_menu()
            with dpg.table(
                tag="root_table",
                header_row=False,
                borders_innerV=True,
                pad_outerX=True,
            ):
                dpg.add_table_column(
                    tag="col_camera",
                    width_fixed=True,
                    init_width_or_weight=_CAMERA_W + 20,
                )
                dpg.add_table_column(tag="col_right")

                with dpg.table_row():
                    self._build_left_panel()
                    self._build_right_panel()

        dpg.set_primary_window("main_window", True)

    def _build_menu(self) -> None:
        with dpg.menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(
                    label="Save Config",
                    callback=self._on_save_config_clicked,
                    shortcut="Ctrl+S",
                )
                dpg.add_separator()
                dpg.add_menu_item(label="Quit", callback=dpg.stop_dearpygui)
            with dpg.menu(label="View"):
                dpg.add_menu_item(
                    label="Refresh Bindings",
                    callback=self._on_refresh_bindings,
                )

    def _build_left_panel(self) -> None:
        with dpg.table_cell():
            dpg.add_image(_TEXTURE_TAG, width=_CAMERA_W, height=_CAMERA_H, tag="camera_image")
            dpg.add_separator()

            # Gesture / confidence status row
            with dpg.group(horizontal=True):
                dpg.add_text("Gesture:", color=_COL_MUTED)
                dpg.add_text("—", tag="lbl_gesture_id", color=_COL_GESTURE)
                dpg.add_spacer(width=16)
                dpg.add_text("Confidence:", color=_COL_MUTED)
                dpg.add_text("—", tag="lbl_confidence", color=_COL_GESTURE)

            dpg.add_separator()
            dpg.add_text("Overlay / Confirmations", color=_COL_MUTED)
            with dpg.child_window(
                tag="overlay_panel",
                height=210,
                border=True,
                no_scrollbar=False,
            ):
                dpg.add_text("", tag="lbl_overlay", wrap=_CAMERA_W - 20)

    def _build_right_panel(self) -> None:
        with dpg.table_cell():
            # Bindings header
            with dpg.group(horizontal=True):
                dpg.add_text("Bindings", color=_COL_HEADER)
                dpg.add_spacer(width=8)
                dpg.add_button(label="Refresh", small=True, callback=self._on_refresh_bindings)
                dpg.add_button(label="Save Config", small=True, callback=self._on_save_config_clicked)

            with dpg.child_window(
                tag="bindings_panel",
                height=385,
                border=False,
                no_scrollbar=False,
            ):
                pass  # table built dynamically inside tick

            dpg.add_separator()
            dpg.add_text("Action Log", color=_COL_HEADER)
            with dpg.child_window(
                tag="log_panel",
                height=200,
                border=True,
                no_scrollbar=False,
            ):
                dpg.add_text("", tag="lbl_log", wrap=520)

    # ── Bindings table ─────────────────────────────────────────────────────────

    def _rebuild_bindings_table(self) -> None:
        if dpg.does_item_exist("bindings_table"):
            dpg.delete_item("bindings_table")

        bindings = self._get_bindings()

        with dpg.table(
            tag="bindings_table",
            parent="bindings_panel",
            header_row=True,
            borders_innerH=True,
            borders_innerV=True,
            borders_outerH=True,
            borders_outerV=True,
            row_background=True,
            scrollY=True,
            height=370,
        ):
            dpg.add_table_column(label="ID",      width_fixed=True, init_width_or_weight=170)
            dpg.add_table_column(label="Gesture", width_fixed=True, init_width_or_weight=100)
            dpg.add_table_column(label="Action",  width_fixed=True, init_width_or_weight=150)
            dpg.add_table_column(label="Trigger", width_fixed=True, init_width_or_weight=80)
            dpg.add_table_column(label="Confirm", width_fixed=True, init_width_or_weight=90)
            dpg.add_table_column(label="On",      width_fixed=True, init_width_or_weight=24)
            dpg.add_table_column(label="⚙",       width_fixed=True, init_width_or_weight=140)

            for b in bindings:
                with dpg.table_row():
                    dpg.add_text(b.id,               color=_COL_MUTED)
                    dpg.add_text(b.gesture_id,        color=_COL_GESTURE_ID)
                    dpg.add_text(b.action_id,         color=_COL_ACTION_ID)
                    dpg.add_text(b.trigger,           color=_COL_MUTED)
                    dpg.add_text(b.confirmation.type, color=_COL_MUTED)
                    dot_color = _COL_ON if b.enabled else _COL_OFF
                    dpg.add_text("●", color=dot_color)
                    with dpg.group(horizontal=True):
                        dpg.add_button(
                            label="On/Off",
                            small=True,
                            callback=self._make_toggle_cb(b.id),
                        )
                        dpg.add_button(
                            label="Edit",
                            small=True,
                            callback=self._make_edit_cb(b.id),
                        )
                        dpg.add_button(
                            label="Del",
                            small=True,
                            callback=self._make_delete_cb(b.id),
                        )

        self._bindings_dirty = False

    def _make_toggle_cb(self, binding_id: str) -> Callable:
        def cb() -> None:
            try:
                self._on_toggle_binding(binding_id)
            except Exception as exc:
                self._push_error(str(exc))
            self._bindings_dirty = True
        return cb

    def _make_edit_cb(self, binding_id: str) -> Callable:
        def cb() -> None:
            self._edit_binding_id = binding_id
            self._open_edit_modal(binding_id)
        return cb

    def _make_delete_cb(self, binding_id: str) -> Callable:
        def cb() -> None:
            try:
                self._on_delete_binding(binding_id)
            except Exception as exc:
                self._push_error(str(exc))
            self._bindings_dirty = True
        return cb

    # ── Edit modal ─────────────────────────────────────────────────────────────

    def _open_edit_modal(self, binding_id: str) -> None:
        bindings = self._get_bindings()
        binding = next((b for b in bindings if b.id == binding_id), None)
        if binding is None:
            return

        if dpg.does_item_exist("edit_modal"):
            dpg.delete_item("edit_modal")

        gesture_ids = self._get_gesture_ids()
        action_ids = self._get_action_ids()

        with dpg.window(
            tag="edit_modal",
            label=f"Edit: {binding_id}",
            modal=True,
            width=420,
            height=340,
            pos=[390, 190],
            no_resize=True,
        ):
            dpg.add_text("Gesture ID:")
            dpg.add_combo(
                gesture_ids,
                default_value=binding.gesture_id,
                tag="edit_gesture_id",
                width=220,
            )

            dpg.add_text("Action ID:")
            dpg.add_combo(
                action_ids,
                default_value=binding.action_id,
                tag="edit_action_id",
                width=220,
            )

            dpg.add_text("Trigger:")
            dpg.add_combo(
                ["on_start", "on_hold", "on_release"],
                default_value=binding.trigger,
                tag="edit_trigger",
                width=160,
            )

            dpg.add_text("Min confidence:")
            dpg.add_slider_float(
                default_value=binding.min_confidence,
                min_value=0.0,
                max_value=1.0,
                tag="edit_confidence",
                width=220,
                format="%.2f",
            )

            dpg.add_text("Cooldown (ms):")
            dpg.add_input_int(
                default_value=binding.cooldown_ms,
                tag="edit_cooldown",
                width=120,
                min_value=0,
                min_clamped=True,
            )

            dpg.add_text("Confirmation type:")
            dpg.add_combo(
                ["none", "hold", "repeat_gesture", "second_gesture", "hotkey"],
                default_value=binding.confirmation.type,
                tag="edit_confirm_type",
                width=200,
            )

            dpg.add_text("Enabled:")
            dpg.add_checkbox(default_value=binding.enabled, tag="edit_enabled")

            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="Save",   callback=self._on_save_edit_clicked)
                dpg.add_button(label="Cancel", callback=lambda: dpg.delete_item("edit_modal"))

            dpg.add_text("", tag="edit_error_lbl", color=_COL_ERROR)

    def _on_save_edit_clicked(self) -> None:
        if self._edit_binding_id is None:
            return

        from confirmation.confirmation_policy import ConfirmationPolicy

        updates: dict[str, Any] = {
            "gesture_id":     dpg.get_value("edit_gesture_id"),
            "action_id":      dpg.get_value("edit_action_id"),
            "trigger":        dpg.get_value("edit_trigger"),
            "min_confidence": round(float(dpg.get_value("edit_confidence")), 2),
            "cooldown_ms":    int(dpg.get_value("edit_cooldown")),
            "enabled":        bool(dpg.get_value("edit_enabled")),
            "confirmation":   ConfirmationPolicy(
                                  type=dpg.get_value("edit_confirm_type")  # type: ignore[arg-type]
                              ),
        }

        try:
            self._on_save_binding(self._edit_binding_id, updates)
            dpg.delete_item("edit_modal")
            self._bindings_dirty = True
        except Exception as exc:
            if dpg.does_item_exist("edit_error_lbl"):
                dpg.set_value("edit_error_lbl", f"Error: {exc}")

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _on_refresh_bindings(self) -> None:
        self._bindings_dirty = True

    def _on_save_config_clicked(self) -> None:
        try:
            self._on_save_config()
            self._push_message("Config saved to bindings.json")
        except Exception as exc:
            self._push_error(str(exc))

    def _on_viewport_resize(self) -> None:
        vw = dpg.get_viewport_width()
        vh = dpg.get_viewport_height()
        dpg.set_item_width("main_window", vw)
        dpg.set_item_height("main_window", vh)

    # ── Overlay helpers ────────────────────────────────────────────────────────

    def _push_message(self, text: str) -> None:
        self._append_overlay(OverlayMessage(kind="message", text=text))

    def _push_error(self, text: str) -> None:
        self._append_overlay(OverlayMessage(kind="error", text=text))

    def _append_overlay(self, msg: OverlayMessage) -> None:
        self._overlay_msgs.append(msg)
        if len(self._overlay_msgs) > _OVERLAY_MAX:
            self._overlay_msgs = self._overlay_msgs[-_OVERLAY_MAX:]
        self._flush_overlay_label()

        # Log "Executing:" messages to the action log
        if msg.kind == "message" and msg.text.startswith("Executing:"):
            preview = msg.text[len("Executing:"):].strip()
            self._action_log.append(ActionLogEntry(
                timestamp=time.time(),
                action_preview=preview,
            ))
            if len(self._action_log) > _LOG_MAX:
                self._action_log = self._action_log[-_LOG_MAX:]
            self._flush_log_label()

    def _flush_overlay_label(self) -> None:
        if not dpg.does_item_exist("lbl_overlay"):
            return
        lines: list[str] = []
        for msg in reversed(self._overlay_msgs):
            if msg.kind == "confirmation":
                lines.append(f"[CONFIRM] {msg.title}")
                lines.append(f"  Action : {msg.text}")
                lines.append(f"  → {msg.instruction}")
            elif msg.kind == "error":
                lines.append(f"[ERROR] {msg.text}")
            else:
                lines.append(f"[INFO]  {msg.text}")
        dpg.set_value("lbl_overlay", "\n".join(lines))

    def _flush_log_label(self) -> None:
        if not dpg.does_item_exist("lbl_log"):
            return
        lines: list[str] = []
        for entry in reversed(self._action_log[-15:]):
            ts = time.strftime("%H:%M:%S", time.localtime(entry.timestamp))
            lines.append(f"[{ts}] {entry.action_preview}")
        dpg.set_value("lbl_log", "\n".join(lines))

    # ── Queue draining ─────────────────────────────────────────────────────────

    def _drain_frame_queue(self) -> None:
        try:
            packet = self._frame_q.get_nowait()
        except Empty:
            return

        dpg.set_value(_TEXTURE_TAG, packet.texture_data)

        if packet.gesture_id:
            dpg.set_value("lbl_gesture_id", packet.gesture_id)
            dpg.set_value("lbl_confidence",  f"{packet.confidence:.2f}")
        elif not packet.has_hands:
            dpg.set_value("lbl_gesture_id", "—")
            dpg.set_value("lbl_confidence",  "—")

    def _drain_gesture_queue(self) -> None:
        try:
            while True:
                event = self._gesture_q.get_nowait()
                dpg.set_value("lbl_gesture_id", event.gesture_id)
                dpg.set_value("lbl_confidence",  f"{event.confidence:.2f}")
                self._on_gesture_event(event)
        except Empty:
            pass

    def _drain_overlay_queue(self) -> None:
        try:
            while True:
                msg = self._overlay_q.get_nowait()
                self._append_overlay(msg)
        except Empty:
            pass

    # ── Main render loop ───────────────────────────────────────────────────────

    def tick(self) -> None:
        """Called once per rendered frame (main thread only)."""
        self._drain_frame_queue()
        self._drain_gesture_queue()
        self._drain_overlay_queue()
        self._on_controller_tick()

        if self._bindings_dirty:
            self._rebuild_bindings_table()

    def run(self) -> None:
        while dpg.is_dearpygui_running():
            self.tick()
            dpg.render_dearpygui_frame()
        dpg.destroy_context()
