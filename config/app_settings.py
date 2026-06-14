from __future__ import annotations
import json
from pathlib import Path
from pydantic import BaseModel, field_validator


_DETECT_PRESETS = {
    "160×120 (fastest)":  (160, 120),
    "320×240 (default)":  (320, 240),
    "426×240":            (426, 240),
    "640×480 (accurate)": (640, 480),
}
DETECT_PRESET_LABELS = list(_DETECT_PRESETS.keys())


class AppSettings(BaseModel):
    camera_index: int = 0
    target_fps: int = 30
    detect_preset: str = "320×240 (default)"
    capture_zone_enabled: bool = True
    capture_zone_size: float = 0.45   # fraction of frame width (both axes)
    skeleton_only: bool = False       # replace camera feed with black background

    @field_validator("detect_preset")
    @classmethod
    def _valid_preset(cls, v: str) -> str:
        if v not in _DETECT_PRESETS:
            return "320×240 (default)"
        return v

    @property
    def detect_size(self) -> tuple[int, int]:
        return _DETECT_PRESETS[self.detect_preset]

    @property
    def frame_interval(self) -> float:
        return 1.0 / max(1, self.target_fps)

    # ── Persistence ────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Path) -> "AppSettings":
        if path.exists():
            try:
                return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                pass
        return cls()

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
