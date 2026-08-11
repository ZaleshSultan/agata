import unittest
from tools.datetime_tool import get_current_datetime, TIMEZONE_NAME


class DatetimeToolTests(unittest.TestCase):

    def test_current_datetime(self) -> None:
        """get_current_datetime() returns non-empty string with day, month, year, time."""
        result = get_current_datetime()
        self.assertIsNotNone(result)
        self.assertTrue(result.startswith("Сейчас "))
        self.assertIn(":", result)

    def test_current_timezone(self) -> None:
        """Verify timezone constant is Asia/Almaty."""
        self.assertEqual(TIMEZONE_NAME, "Asia/Almaty")


if __name__ == "__main__":
    unittest.main()
