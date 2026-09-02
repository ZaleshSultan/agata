"""Screen capture only; vision inference belongs to a provider implementation."""

from __future__ import annotations

import io

import config


class ScreenCaptureError(RuntimeError):
    """A user-facing, provider-independent capture failure."""


def capture_screen() -> bytes:
    """Capture all visible screens as PNG bytes suitable for a vision provider."""
    if not config.VISION_ENABLED:
        raise ScreenCaptureError("Vision is disabled by VISION_ENABLED.")
    try:
        from PIL import ImageGrab

        image = ImageGrab.grab(all_screens=True)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    except ImportError as exc:
        raise ScreenCaptureError("Pillow is not installed. Run: pip install -r requirements.txt") from exc
    except Exception as exc:
        raise ScreenCaptureError(f"Screen capture failed: {exc}") from exc
