"""Screen capture is tested without reading the user's real desktop."""

from io import BytesIO
from unittest.mock import patch

from PIL import Image

from vision.screen import capture_screen


def test_capture_screen_returns_png_bytes() -> None:
    image = Image.new("RGB", (4, 3), "navy")
    with patch("PIL.ImageGrab.grab", return_value=image) as grab:
        result = capture_screen()
    grab.assert_called_once_with(all_screens=True)
    assert result.startswith(b"\x89PNG\r\n\x1a\n")
    restored = Image.open(BytesIO(result))
    assert restored.size == (4, 3)
