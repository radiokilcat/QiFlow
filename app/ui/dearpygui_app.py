from __future__ import annotations
import json
import time
import uuid
from queue import Queue, Empty
from typing import Callable, Any

import numpy as np
import dearpygui.dearpygui as dpg

from gestures.base import GestureEvent
from .view_models import FramePacket, OverlayMessage, ActionLogEntry
from config.app_settings import AppSettings, DETECT_PRESET_LABELS

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

# ── Pose diagram ───────────────────────────────────────────────────────────────
_DIAG_SIZE = 130  # px — square drawlist for hand pose preview

_DIAG_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Normalised (x, y) for each of the 21 hand landmarks per static pose.
# x: 0=left … 1=right,  y: 0=top … 1=bottom  (right hand, palm facing viewer)
_POSE_LANDMARKS: dict[str, list[tuple[float, float]]] = {
    "open_palm": [
        (0.43, 0.95),  # 0  wrist
        (0.23, 0.83),  # 1  thumb CMC
        (0.10, 0.72),  # 2  thumb MCP
        (0.05, 0.57),  # 3  thumb IP
        (0.08, 0.43),  # 4  thumb TIP
        (0.36, 0.63),  # 5  index MCP
        (0.31, 0.47),  # 6  index PIP
        (0.28, 0.33),  # 7  index DIP
        (0.27, 0.19),  # 8  index TIP
        (0.48, 0.61),  # 9  middle MCP
        (0.47, 0.44),  # 10 middle PIP
        (0.47, 0.30),  # 11 middle DIP
        (0.47, 0.16),  # 12 middle TIP
        (0.59, 0.63),  # 13 ring MCP
        (0.62, 0.47),  # 14 ring PIP
        (0.63, 0.33),  # 15 ring DIP
        (0.64, 0.20),  # 16 ring TIP
        (0.71, 0.70),  # 17 pinky MCP
        (0.75, 0.57),  # 18 pinky PIP
        (0.78, 0.46),  # 19 pinky DIP
        (0.80, 0.36),  # 20 pinky TIP
    ],
    "fist": [
        (0.43, 0.95),  # 0  wrist
        (0.27, 0.82),  # 1  thumb CMC
        (0.18, 0.72),  # 2  thumb MCP
        (0.17, 0.63),  # 3  thumb IP
        (0.23, 0.59),  # 4  thumb TIP
        (0.38, 0.65),  # 5  index MCP
        (0.38, 0.74),  # 6  index PIP
        (0.39, 0.81),  # 7  index DIP
        (0.40, 0.86),  # 8  index TIP
        (0.48, 0.63),  # 9  middle MCP
        (0.49, 0.73),  # 10 middle PIP
        (0.50, 0.80),  # 11 middle DIP
        (0.50, 0.85),  # 12 middle TIP
        (0.59, 0.64),  # 13 ring MCP
        (0.60, 0.73),  # 14 ring PIP
        (0.61, 0.79),  # 15 ring DIP
        (0.61, 0.84),  # 16 ring TIP
        (0.69, 0.70),  # 17 pinky MCP
        (0.71, 0.77),  # 18 pinky PIP
        (0.72, 0.82),  # 19 pinky DIP
        (0.72, 0.87),  # 20 pinky TIP
    ],
    "pinch": [
        (0.43, 0.95),  # 0  wrist
        (0.25, 0.83),  # 1  thumb CMC
        (0.13, 0.72),  # 2  thumb MCP
        (0.11, 0.58),  # 3  thumb IP
        (0.33, 0.41),  # 4  thumb TIP  ← meets index
        (0.40, 0.64),  # 5  index MCP
        (0.37, 0.52),  # 6  index PIP
        (0.35, 0.44),  # 7  index DIP
        (0.33, 0.41),  # 8  index TIP  ← meets thumb
        (0.51, 0.62),  # 9  middle MCP
        (0.53, 0.72),  # 10 middle PIP
        (0.54, 0.79),  # 11 middle DIP
        (0.54, 0.85),  # 12 middle TIP
        (0.61, 0.64),  # 13 ring MCP
        (0.63, 0.73),  # 14 ring PIP
        (0.64, 0.80),  # 15 ring DIP
        (0.64, 0.85),  # 16 ring TIP
        (0.70, 0.70),  # 17 pinky MCP
        (0.72, 0.77),  # 18 pinky PIP
        (0.73, 0.82),  # 19 pinky DIP
        (0.73, 0.87),  # 20 pinky TIP
    ],
    "point": [
        (0.43, 0.95),  # 0  wrist
        (0.38, 0.86),  # 1  thumb CMC   ← shifts right toward thumb side
        (0.55, 0.80),  # 2  thumb MCP   ← going right
        (0.70, 0.75),  # 3  thumb IP    ← further right
        (0.84, 0.71),  # 4  thumb TIP   ← rightmost, nearly horizontal
        (0.36, 0.65),  # 5  index MCP
        (0.31, 0.49),  # 6  index PIP
        (0.28, 0.34),  # 7  index DIP
        (0.26, 0.19),  # 8  index TIP   ← extended up
        (0.48, 0.63),  # 9  middle MCP
        (0.49, 0.73),  # 10 middle PIP
        (0.50, 0.80),  # 11 middle DIP
        (0.50, 0.85),  # 12 middle TIP  (curled)
        (0.58, 0.64),  # 13 ring MCP
        (0.59, 0.73),  # 14 ring PIP
        (0.60, 0.79),  # 15 ring DIP
        (0.60, 0.84),  # 16 ring TIP
        (0.67, 0.70),  # 17 pinky MCP
        (0.69, 0.77),  # 18 pinky PIP
        (0.70, 0.82),  # 19 pinky DIP
        (0.70, 0.87),  # 20 pinky TIP
    ],
}


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
        on_add_binding: Callable[[dict[str, Any]], None],
        on_save_binding: Callable[[str, dict[str, Any]], None],
        on_toggle_binding: Callable[[str], None],
        on_delete_binding: Callable[[str], None],
        on_save_config: Callable[[], None],
        get_bindings: Callable[[], list],
        get_action_ids: Callable[[], list[str]],
        get_gesture_ids: Callable[[], list[str]],
        preview_action: Callable[[str, dict[str, Any]], str],
        settings: AppSettings,
        on_save_settings: Callable[[dict[str, Any]], None],
        on_restart_camera: Callable[[int], None],
        get_movements: Callable[[], list[dict[str, Any]]],
        save_movement: Callable[[str, dict[str, Any]], None],
        get_gestures: Callable[[], list[dict[str, Any]]],
        save_gestures: Callable[[list[dict[str, Any]]], None],
    ) -> None:
        self._frame_q = frame_queue
        self._gesture_q = gesture_queue
        self._overlay_q = overlay_queue

        self._settings = settings
        self._on_save_settings = on_save_settings
        self._on_restart_camera = on_restart_camera
        self._on_gesture_event = on_gesture_event
        self._on_controller_tick = on_controller_tick
        self._on_add_binding = on_add_binding
        self._on_save_binding = on_save_binding
        self._on_toggle_binding = on_toggle_binding
        self._on_delete_binding = on_delete_binding
        self._on_save_config = on_save_config
        self._get_bindings = get_bindings
        self._get_action_ids = get_action_ids
        self._get_gesture_ids = get_gesture_ids
        self._preview_action = preview_action

        self._get_movements = get_movements
        self._save_movement = save_movement
        self._get_gestures = get_gestures
        self._save_gestures = save_gestures

        self._overlay_msgs: list[OverlayMessage] = []
        self._action_log: list[ActionLogEntry] = []
        self._edit_binding_id: str | None = None
        self._is_new_binding = False
        self._bindings_dirty = True   # rebuild table on first tick
        self._mv_data: dict[str, dict[str, Any]] = {}

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
                dpg.add_menu_item(label="Settings…", callback=self._open_settings_modal)
                dpg.add_separator()
                dpg.add_menu_item(label="Quit", callback=dpg.stop_dearpygui)
            with dpg.menu(label="View"):
                dpg.add_menu_item(
                    label="Refresh Bindings",
                    callback=self._on_refresh_bindings,
                )
            with dpg.menu(label="Edit"):
                dpg.add_menu_item(
                    label="Movement Editor…",
                    callback=self._open_movement_editor,
                )
                dpg.add_menu_item(
                    label="Gestures Editor…",
                    callback=self._open_gestures_editor,
                )

    def _build_left_panel(self) -> None:
        with dpg.table_cell():
            dpg.add_image(_TEXTURE_TAG, width=_CAMERA_W, height=_CAMERA_H, tag="camera_image")
            dpg.add_separator()

            # Gesture / hand / confidence status row
            with dpg.group(horizontal=True):
                dpg.add_text("Gesture:", color=_COL_MUTED)
                dpg.add_text("—", tag="lbl_gesture_id", color=_COL_GESTURE)
                dpg.add_spacer(width=16)
                dpg.add_text("Hand:", color=_COL_MUTED)
                dpg.add_text("—", tag="lbl_hand_side", color=_COL_GESTURE_ID)
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
                dpg.add_button(label="+ Add", small=True, callback=self._on_add_binding_clicked)
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
            self._is_new_binding = False
            self._edit_binding_id = binding_id
            self._open_edit_modal(binding_id)
        return cb

    def _on_add_binding_clicked(self) -> None:
        self._is_new_binding = True
        self._edit_binding_id = "binding_" + uuid.uuid4().hex[:8]
        self._open_edit_modal(None)

    def _make_delete_cb(self, binding_id: str) -> Callable:
        def cb() -> None:
            try:
                self._on_delete_binding(binding_id)
            except Exception as exc:
                self._push_error(str(exc))
            self._bindings_dirty = True
        return cb

    # ── Edit / Add modal ───────────────────────────────────────────────────────

    def _open_edit_modal(self, binding_id: str | None) -> None:
        if dpg.does_item_exist("edit_modal"):
            dpg.delete_item("edit_modal")

        gesture_ids = self._get_gesture_ids()
        action_ids  = self._get_action_ids()

        if self._is_new_binding:
            # Defaults for a brand-new binding
            g_id     = gesture_ids[0] if gesture_ids else ""
            a_id     = action_ids[0]  if action_ids  else ""
            trigger  = "on_start"
            conf     = 0.75
            cool     = 500
            params   = "{}"
            ctype    = "none"
            hold_ms  = 700
            t_ms     = 1500
            excl     = False
            enabled  = True
            title    = f"Add binding: {self._edit_binding_id}"
        else:
            bindings = self._get_bindings()
            b = next((b for b in bindings if b.id == binding_id), None)
            if b is None:
                return
            g_id     = b.gesture_id
            a_id     = b.action_id
            trigger  = b.trigger
            conf     = b.min_confidence
            cool     = b.cooldown_ms
            params   = json.dumps(b.action_params)
            ctype    = b.confirmation.type
            hold_ms  = b.confirmation.hold_ms
            t_ms     = b.confirmation.timeout_ms
            excl     = b.exclusive
            enabled  = b.enabled
            title    = f"Edit: {binding_id}"

        with dpg.window(
            tag="edit_modal",
            label=title,
            modal=True,
            width=460,
            height=560,
            pos=[370, 80],
            no_resize=False,
        ):
            dpg.add_text("Gesture ID:")
            dpg.add_combo(gesture_ids, default_value=g_id,    tag="edit_gesture_id", width=240)

            dpg.add_text("Action ID:")
            dpg.add_combo(action_ids,  default_value=a_id,    tag="edit_action_id",  width=240)

            dpg.add_text("Action params (JSON):")
            dpg.add_input_text(
                default_value=params,
                tag="edit_action_params",
                width=420,
                height=56,
                multiline=True,
            )
            with dpg.group(horizontal=True):
                dpg.add_button(label="Preview", small=True, callback=self._on_preview_clicked)
                dpg.add_text("", tag="edit_preview_lbl", color=_COL_ACTION_ID)

            dpg.add_separator()

            dpg.add_text("Trigger:")
            dpg.add_combo(
                ["on_start", "on_hold", "on_release"],
                default_value=trigger,
                tag="edit_trigger",
                width=160,
            )

            dpg.add_text("Min confidence:")
            dpg.add_slider_float(
                default_value=conf, min_value=0.0, max_value=1.0,
                tag="edit_confidence", width=240, format="%.2f",
            )

            dpg.add_text("Cooldown (ms):")
            dpg.add_input_int(
                default_value=cool, tag="edit_cooldown",
                width=120, min_value=0, min_clamped=True,
            )

            dpg.add_separator()

            dpg.add_text("Confirmation type:")
            dpg.add_combo(
                ["none", "hold", "repeat_gesture", "second_gesture", "hotkey"],
                default_value=ctype, tag="edit_confirm_type", width=200,
            )

            with dpg.group(horizontal=True):
                dpg.add_text("Hold (ms):")
                dpg.add_input_int(
                    default_value=hold_ms, tag="edit_hold_ms",
                    width=100, min_value=0, min_clamped=True,
                )
                dpg.add_spacer(width=12)
                dpg.add_text("Timeout (ms):")
                dpg.add_input_int(
                    default_value=t_ms, tag="edit_timeout_ms",
                    width=100, min_value=0, min_clamped=True,
                )

            dpg.add_separator()

            with dpg.group(horizontal=True):
                dpg.add_text("Enabled:")
                dpg.add_checkbox(default_value=enabled, tag="edit_enabled")
                dpg.add_spacer(width=20)
                dpg.add_text("Exclusive:")
                dpg.add_checkbox(default_value=excl, tag="edit_exclusive")

            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="Save",   callback=self._on_save_edit_clicked)
                dpg.add_button(label="Cancel", callback=lambda: dpg.delete_item("edit_modal"))

            dpg.add_text("", tag="edit_error_lbl", color=_COL_ERROR)

    def _parse_modal_fields(self) -> dict[str, Any] | str:
        """Parse all modal fields; return dict on success or error string."""
        from confirmation.confirmation_policy import ConfirmationPolicy
        try:
            action_params = json.loads(dpg.get_value("edit_action_params") or "{}")
        except json.JSONDecodeError as exc:
            return f"action_params JSON error: {exc}"
        if not isinstance(action_params, dict):
            return "action_params must be a JSON object {…}"
        return {
            "gesture_id":     dpg.get_value("edit_gesture_id"),
            "action_id":      dpg.get_value("edit_action_id"),
            "trigger":        dpg.get_value("edit_trigger"),
            "min_confidence": round(float(dpg.get_value("edit_confidence")), 2),
            "cooldown_ms":    int(dpg.get_value("edit_cooldown")),
            "action_params":  action_params,
            "enabled":        bool(dpg.get_value("edit_enabled")),
            "exclusive":      bool(dpg.get_value("edit_exclusive")),
            "confirmation":   ConfirmationPolicy(
                type=dpg.get_value("edit_confirm_type"),  # type: ignore[arg-type]
                hold_ms=int(dpg.get_value("edit_hold_ms")),
                timeout_ms=int(dpg.get_value("edit_timeout_ms")),
            ),
        }

    def _on_preview_clicked(self) -> None:
        fields = self._parse_modal_fields()
        if isinstance(fields, str):
            if dpg.does_item_exist("edit_preview_lbl"):
                dpg.set_value("edit_preview_lbl", fields)
            return
        preview = self._preview_action(fields["action_id"], fields["action_params"])
        if dpg.does_item_exist("edit_preview_lbl"):
            dpg.set_value("edit_preview_lbl", preview)

    def _on_save_edit_clicked(self) -> None:
        if self._edit_binding_id is None:
            return
        fields = self._parse_modal_fields()
        if isinstance(fields, str):
            if dpg.does_item_exist("edit_error_lbl"):
                dpg.set_value("edit_error_lbl", fields)
            return
        try:
            if self._is_new_binding:
                self._on_add_binding({"id": self._edit_binding_id, **fields})
            else:
                self._on_save_binding(self._edit_binding_id, fields)
            dpg.delete_item("edit_modal")
            self._bindings_dirty = True
        except Exception as exc:
            if dpg.does_item_exist("edit_error_lbl"):
                dpg.set_value("edit_error_lbl", f"Error: {exc}")

    # ── Movement Editor ────────────────────────────────────────────────────────

    def _open_movement_editor(self) -> None:
        if dpg.does_item_exist("movement_editor"):
            dpg.focus_item("movement_editor")
            return

        movements = self._get_movements()
        self._mv_data = {m.get("type", ""): m for m in movements}
        mtypes = [m.get("type", "") for m in movements]
        initial = mtypes[0] if mtypes else ""

        with dpg.window(
            tag="movement_editor",
            label="Movement Editor",
            width=420,
            height=260,
            pos=[300, 120],
            no_resize=False,
            on_close=lambda: dpg.delete_item("movement_editor"),
        ):
            with dpg.group(horizontal=True):
                dpg.add_text("Movement:", color=_COL_MUTED)
                dpg.add_combo(
                    mtypes,
                    default_value=initial,
                    tag="mv_selector",
                    width=200,
                    callback=lambda _, a: self._mv_on_select(a),
                )
            dpg.add_separator()
            with dpg.group(tag="mv_fields_group"):
                pass
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="Save", callback=self._mv_on_save)
                dpg.add_text("", tag="mv_status_lbl", color=_COL_GESTURE)
            dpg.add_text("", tag="mv_error_lbl", color=_COL_ERROR)

        if initial:
            self._mv_rebuild_fields(initial)

    def _mv_on_select(self, value: str) -> None:
        self._mv_rebuild_fields(value)
        if dpg.does_item_exist("mv_status_lbl"):
            dpg.set_value("mv_status_lbl", "")
        if dpg.does_item_exist("mv_error_lbl"):
            dpg.set_value("mv_error_lbl", "")

    def _mv_rebuild_fields(self, mtype: str) -> None:
        if not dpg.does_item_exist("mv_fields_group"):
            return
        dpg.delete_item("mv_fields_group", children_only=True)
        m = self._mv_data.get(mtype, {})
        if mtype == "mouse_track":
            pose = m.get("activator_pose", "fist")
            with dpg.group(horizontal=True, parent="mv_fields_group"):
                with dpg.group():
                    dpg.add_text("Activator hand:", color=_COL_MUTED)
                    dpg.add_combo(
                        ["left", "right"],
                        default_value=m.get("activator_hand", "left"),
                        tag="mv_activator_hand",
                        width=160,
                    )
                    dpg.add_spacer(height=4)
                    dpg.add_text("Activator pose:", color=_COL_MUTED)
                    dpg.add_combo(
                        ["fist", "open_palm", "pinch", "point"],
                        default_value=pose,
                        tag="mv_activator_pose",
                        width=160,
                        callback=lambda _, a: self._mv_update_diagram(a),
                    )
                    dpg.add_spacer(height=4)
                    dpg.add_text("Activation type:", color=_COL_MUTED)
                    dpg.add_combo(
                        ["constant", "toggle"],
                        default_value=m.get("activation_type", "constant"),
                        tag="mv_activation_type",
                        width=160,
                    )
                dpg.add_spacer(width=10)
                dpg.add_drawlist(
                    tag="mv_pose_diagram",
                    width=_DIAG_SIZE,
                    height=_DIAG_SIZE,
                )
            self._mv_draw_pose(pose)
        else:
            dpg.add_text(
                "No configurable options for this movement type.",
                parent="mv_fields_group",
                color=_COL_MUTED,
            )

    def _mv_update_diagram(self, pose: str) -> None:
        if not dpg.does_item_exist("mv_pose_diagram"):
            return
        dpg.delete_item("mv_pose_diagram", children_only=True)
        self._mv_draw_pose(pose)

    def _mv_draw_pose(self, pose: str) -> None:
        if not dpg.does_item_exist("mv_pose_diagram"):
            return
        lms = _POSE_LANDMARKS.get(pose)
        if lms is None:
            return
        s = _DIAG_SIZE
        # Background
        dpg.draw_rectangle(
            (0, 0), (s - 1, s - 1),
            color=(55, 55, 55, 255),
            fill=(18, 18, 18, 255),
            parent="mv_pose_diagram",
        )
        pts = [(x * s, y * s) for x, y in lms]
        for a, b in _DIAG_CONNECTIONS:
            dpg.draw_line(
                pts[a], pts[b],
                color=(0, 190, 75, 255),
                thickness=1.5,
                parent="mv_pose_diagram",
            )
        for x, y in pts:
            dpg.draw_circle(
                (x, y), 3.5,
                color=(0, 130, 50, 255),
                fill=(235, 235, 235, 255),
                parent="mv_pose_diagram",
            )

    def _mv_on_save(self) -> None:
        if not dpg.does_item_exist("mv_selector"):
            return
        mtype = dpg.get_value("mv_selector")
        fields: dict[str, Any] = {}
        if mtype == "mouse_track":
            fields["activator_hand"]   = dpg.get_value("mv_activator_hand")
            fields["activator_pose"]   = dpg.get_value("mv_activator_pose")
            fields["activation_type"]  = dpg.get_value("mv_activation_type")
        try:
            self._save_movement(mtype, fields)
            self._mv_data[mtype] = {**self._mv_data.get(mtype, {}), **fields}
            if dpg.does_item_exist("mv_status_lbl"):
                dpg.set_value("mv_status_lbl", "Saved")
            if dpg.does_item_exist("mv_error_lbl"):
                dpg.set_value("mv_error_lbl", "")
        except Exception as exc:
            if dpg.does_item_exist("mv_error_lbl"):
                dpg.set_value("mv_error_lbl", f"Error: {exc}")

    # ── Gestures Editor ────────────────────────────────────────────────────────

    _GESTURE_NAMES: dict[str, str] = {
        "fist": "Fist",
        "open_palm": "Open Palm",
        "pinch": "Pinch",
        "point": "Point",
    }

    def _open_gestures_editor(self) -> None:
        if dpg.does_item_exist("gestures_editor"):
            dpg.focus_item("gestures_editor")
            return

        poses = self._get_gestures()

        with dpg.window(
            tag="gestures_editor",
            label="Gestures Editor",
            width=340,
            height=40 + 32 * len(poses) + 70,
            pos=[320, 140],
            no_resize=False,
            on_close=lambda: dpg.delete_item("gestures_editor"),
        ):
            with dpg.table(
                header_row=True,
                borders_innerH=True,
                borders_outerH=True,
                borders_outerV=True,
                row_background=True,
            ):
                dpg.add_table_column(label="Gesture",  width_fixed=True, init_width_or_weight=120)
                dpg.add_table_column(label="Hand",     width_fixed=True, init_width_or_weight=70)
                dpg.add_table_column(label="Active",   width_fixed=True, init_width_or_weight=50)

                for i, pose in enumerate(poses):
                    with dpg.table_row():
                        name = self._GESTURE_NAMES.get(pose["type"], pose["type"])
                        dpg.add_text(name, color=_COL_GESTURE_ID)
                        dpg.add_text(pose.get("hand", "any"), color=_COL_MUTED)
                        dpg.add_checkbox(
                            default_value=pose.get("enabled", True),
                            tag=f"gest_enabled_{i}",
                        )

            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="Save", callback=self._ge_on_save)
                dpg.add_text("", tag="ge_status_lbl", color=_COL_GESTURE)
            dpg.add_text("", tag="ge_error_lbl", color=_COL_ERROR)

    def _ge_on_save(self) -> None:
        poses = self._get_gestures()
        updates: list[dict[str, Any]] = []
        for i, pose in enumerate(poses):
            tag = f"gest_enabled_{i}"
            if not dpg.does_item_exist(tag):
                continue
            updates.append({
                "type": pose["type"],
                "hand": pose.get("hand", "any"),
                "enabled": bool(dpg.get_value(tag)),
            })
        try:
            self._save_gestures(updates)
            if dpg.does_item_exist("ge_status_lbl"):
                dpg.set_value("ge_status_lbl", "Saved")
            if dpg.does_item_exist("ge_error_lbl"):
                dpg.set_value("ge_error_lbl", "")
        except Exception as exc:
            if dpg.does_item_exist("ge_error_lbl"):
                dpg.set_value("ge_error_lbl", f"Error: {exc}")

    # ── Settings modal ─────────────────────────────────────────────────────────

    def _open_settings_modal(self) -> None:
        if dpg.does_item_exist("settings_modal"):
            dpg.delete_item("settings_modal")

        s = self._settings
        with dpg.window(
            tag="settings_modal",
            label="Settings",
            modal=True,
            width=360,
            height=310,
            pos=[420, 220],
            no_resize=False,
        ):
            dpg.add_text("Camera", color=_COL_HEADER)
            with dpg.group(horizontal=True):
                dpg.add_text("Camera index:")
                dpg.add_input_int(
                    default_value=s.camera_index,
                    tag="cfg_camera_index",
                    width=70,
                    min_value=0,
                    min_clamped=True,
                )
                dpg.add_button(
                    label="Restart",
                    small=True,
                    callback=self._on_restart_camera_clicked,
                )

            dpg.add_separator()
            dpg.add_text("Performance", color=_COL_HEADER)

            with dpg.group(horizontal=True):
                dpg.add_text("Target FPS:")
                dpg.add_slider_int(
                    default_value=s.target_fps,
                    tag="cfg_fps",
                    min_value=10,
                    max_value=60,
                    width=180,
                )

            dpg.add_text("Detection resolution:")
            dpg.add_combo(
                DETECT_PRESET_LABELS,
                default_value=s.detect_preset,
                tag="cfg_detect_preset",
                width=240,
            )

            with dpg.group(horizontal=True):
                dpg.add_text("Skeleton only:")
                dpg.add_checkbox(default_value=s.skeleton_only, tag="cfg_skeleton_only")

            dpg.add_separator()
            dpg.add_text("Capture zone", color=_COL_HEADER)

            with dpg.group(horizontal=True):
                dpg.add_text("Enabled:")
                dpg.add_checkbox(default_value=s.capture_zone_enabled, tag="cfg_zone_enabled")

            with dpg.group(horizontal=True):
                dpg.add_text("Size (% of frame):")
                dpg.add_slider_float(
                    default_value=s.capture_zone_size,
                    tag="cfg_zone_size",
                    min_value=0.15,
                    max_value=0.85,
                    width=160,
                    format="%.2f",
                )

            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="Save", callback=self._on_save_settings_clicked)
                dpg.add_button(label="Close", callback=lambda: dpg.delete_item("settings_modal"))

            dpg.add_text("", tag="cfg_status_lbl", color=_COL_GESTURE)

    def _on_restart_camera_clicked(self) -> None:
        idx = int(dpg.get_value("cfg_camera_index"))
        try:
            self._on_restart_camera(idx)
            if dpg.does_item_exist("cfg_status_lbl"):
                dpg.set_value("cfg_status_lbl", f"Camera {idx} restarted")
        except Exception as exc:
            if dpg.does_item_exist("cfg_status_lbl"):
                dpg.set_value("cfg_status_lbl", f"Error: {exc}")

    def _on_save_settings_clicked(self) -> None:
        try:
            self._on_save_settings({
                "camera_index":        int(dpg.get_value("cfg_camera_index")),
                "target_fps":          int(dpg.get_value("cfg_fps")),
                "detect_preset":       dpg.get_value("cfg_detect_preset"),
                "capture_zone_enabled": bool(dpg.get_value("cfg_zone_enabled")),
                "capture_zone_size":    round(float(dpg.get_value("cfg_zone_size")), 2),
                "skeleton_only":        bool(dpg.get_value("cfg_skeleton_only")),
            })
            if dpg.does_item_exist("cfg_status_lbl"):
                dpg.set_value("cfg_status_lbl", "Saved")
        except Exception as exc:
            if dpg.does_item_exist("cfg_status_lbl"):
                dpg.set_value("cfg_status_lbl", f"Error: {exc}")

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
            dpg.set_value("lbl_hand_side",  packet.hand_side or "—")
            dpg.set_value("lbl_confidence",  f"{packet.confidence:.2f}")
        elif not packet.has_hands:
            dpg.set_value("lbl_gesture_id", "—")
            dpg.set_value("lbl_hand_side",  "—")
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
