"""System volume control with defensive argument normalization."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_STEP = 10
MIN_VOLUME = 0
MAX_VOLUME = 100


def _tool_result(success: bool, **details: Any) -> str:
    return json.dumps({"success": success, **details}, ensure_ascii=False)


def _to_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.strip().rstrip("%").replace(",", ".")
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} должен быть числом от {MIN_VOLUME} до {MAX_VOLUME}") from exc


def _clamp_percent(value: int) -> int:
    return max(MIN_VOLUME, min(MAX_VOLUME, value))


def _get_endpoint_volume():
    """Return a pycaw endpoint volume object across pycaw API versions."""
    try:
        from ctypes import POINTER, cast

        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    except Exception as exc:  # pragma: no cover - depends on local Windows packages
        raise RuntimeError(
            "Управление громкостью требует установленных pycaw и comtypes."
        ) from exc

    speakers = AudioUtilities.GetSpeakers()
    if hasattr(speakers, "EndpointVolume"):
        return speakers.EndpointVolume
    if hasattr(speakers, "Activate"):
        interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(interface, POINTER(IAudioEndpointVolume))
    raise RuntimeError("Не удалось получить интерфейс IAudioEndpointVolume из устройств вывода.")


def _current_volume_percent(endpoint_volume: Any) -> int:
    return round(float(endpoint_volume.GetMasterVolumeLevelScalar()) * 100)


def _set_volume_percent(endpoint_volume: Any, value: int) -> None:
    endpoint_volume.SetMasterVolumeLevelScalar(_clamp_percent(value) / 100, None)


def control_volume(
    payload: dict[str, Any] | None = None,
    *,
    action: str | None = None,
    level: int | str | float | None = None,
    step: int | str | float | None = None,
) -> str:
    """
    Set, increase, or decrease system volume.

    Accepts either direct keyword arguments from the tool registry or a single
    payload dict for compatibility with older calls like
    control_volume({"action": "volume_set", "level": "30"}).
    """
    if payload is not None:
        action = payload.get("action", action)
        level = payload.get("level", level)
        step = payload.get("step", step)

    try:
        if level is not None:
            level = _to_int(level, "level")
        if step is not None:
            step = _to_int(step, "step")

        if action not in {"volume_set", "volume_up", "volume_down"}:
            return _tool_result(False, error=f"Неизвестное действие громкости: {action}")

        if level is not None and not MIN_VOLUME <= level <= MAX_VOLUME:
            return _tool_result(False, error=f"level должен быть от {MIN_VOLUME} до {MAX_VOLUME}")
        if step is not None and not MIN_VOLUME <= step <= MAX_VOLUME:
            return _tool_result(False, error=f"step должен быть от {MIN_VOLUME} до {MAX_VOLUME}")

        endpoint_volume = _get_endpoint_volume()
        current = _current_volume_percent(endpoint_volume)

        if action == "volume_set":
            if level is None:
                return _tool_result(False, error="Для volume_set нужен аргумент level.")
            target = level
        elif action == "volume_up":
            target = current + (step if step is not None else DEFAULT_STEP)
        else:
            target = current - (step if step is not None else DEFAULT_STEP)

        target = _clamp_percent(int(target))
        _set_volume_percent(endpoint_volume, target)
        logger.info("[TOOL] volume action=%s current=%s target=%s", action, current, target)
        return _tool_result(True, action=action, previous=current, volume=target)
    except Exception as exc:
        logger.error("Error adjusting volume: %s", exc, exc_info=True)
        return _tool_result(False, error=f"Не удалось изменить громкость: {exc}")


TOOL_DEFINITION: dict = {
    "function": {
        "name": "control_volume",
        "description": "Установить, увеличить или уменьшить системную громкость в процентах.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["volume_set", "volume_up", "volume_down"],
                    "description": "Действие: установить, увеличить или уменьшить громкость.",
                },
                "level": {
                    "type": "integer",
                    "description": "Абсолютный уровень громкости 0-100 для volume_set.",
                },
                "step": {
                    "type": "integer",
                    "description": "Шаг изменения громкости 0-100 для volume_up/volume_down.",
                },
            },
            "required": ["action"],
        },
    },
    "requires_confirmation": False,
}
