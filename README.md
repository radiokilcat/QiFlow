# Gesture Control

A Python framework for controlling Windows via hand gestures detected through MediaPipe.

## Architecture

```
CameraSource → LandmarkDetector → GestureRecognizers
    → GestureEvent → BindingEngine → ConfirmationManager → ActionDispatcher → OSAdapter
```

### Layers

| Layer | Responsibility |
|---|---|
| `camera/` | Camera access, frame capture |
| `vision/` | MediaPipe wrapper, landmark normalization |
| `gestures/` | Gesture recognizers, GestureEvent production |
| `bindings/` | Binding lookup, cooldown, trigger matching |
| `confirmation/` | Confirmation policies (hold, repeat, hotkey) |
| `actions/` | Action definitions, Pydantic param models |
| `os_control/` | OS operations (stub → real impl) |
| `ui/` | Overlay interface (console → GUI) |
| `config/` | `bindings.json` — all bindings stored here |

### Design rules
- `GestureRecognizer` has no knowledge of `Action`
- `Action` has no knowledge of `Gesture`
- `Binding` never executes OS commands directly
- All OS calls flow through `OSAdapter`
- All action calls flow through `ActionDispatcher`
- Action params are validated via Pydantic models

## Running

Install dependencies:
```bash
pip install -r requirements.txt
```

GUI mode (camera + DearPyGui interface):
```bash
python -m app.main --gui
```

Select a specific camera:
```bash
python -m app.main --gui --camera=1
```

Demo mode (no camera, simulated gesture events):
```bash
python -m app.main
```

## Running tests

```bash
pytest tests/
```

## Adding a new gesture

1. Create `gestures/recognizers/my_gesture.py` inheriting `GestureRecognizer`
2. Register it in `app/main.py`
3. Add a binding in `config/bindings.json`

## Adding a new action

1. Create or extend an `actions/*.py` file inheriting `Action`
2. Define a Pydantic `params_model`
3. Register it in `app/main.py`
