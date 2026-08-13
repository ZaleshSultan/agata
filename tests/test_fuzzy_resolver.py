"""
Unit tests for fuzzy_resolver.py
"""

import unittest
from fuzzy_resolver import (
    calculate_similarity,
    fuzzy_resolve,
    is_fuzzy_match,
    levenshtein_distance,
)


class TestFuzzyResolver(unittest.TestCase):

    def test_levenshtein_distance(self) -> None:
        self.assertEqual(levenshtein_distance("хром", "хром"), 0)
        self.assertEqual(levenshtein_distance("хром", "хрому"), 1)
        self.assertEqual(levenshtein_distance("телеграм", "телеграмму"), 2)
        self.assertEqual(levenshtein_distance("споти", "спотик"), 1)
        self.assertEqual(levenshtein_distance("zoom", "хром"), 4)

    def test_calculate_similarity(self) -> None:
        score, dist, norm_dist = calculate_similarity("chrome", "chrome")
        self.assertEqual(score, 1.0)
        self.assertEqual(dist, 0)

        score, dist, norm_dist = calculate_similarity("телеграм", "телеграмму")
        self.assertGreater(score, 0.75)
        self.assertEqual(dist, 2)

    def test_length_adaptive_matching(self) -> None:
        # Short strings: strict
        matched, score, dist, _ = is_fuzzy_match("вк", "вк")
        self.assertTrue(matched)

        matched, _, _, _ = is_fuzzy_match("вк", "тг")
        self.assertFalse(matched)

        # Chrome vs Zoom: must NOT match
        matched, _, _, _ = is_fuzzy_match("хром", "zoom")
        self.assertFalse(matched)

        # Chrome vs Хрому: must match
        matched, _, _, _ = is_fuzzy_match("хрому", "хром")
        self.assertTrue(matched)

        # Telegram vs Телеграмму: must match
        matched, _, _, _ = is_fuzzy_match("телеграмму", "телеграм")
        self.assertTrue(matched)

        # Spotify vs Спотик: must match
        matched, _, _, _ = is_fuzzy_match("спотик", "споти")
        self.assertTrue(matched)

    def test_fuzzy_resolve_dictionary(self) -> None:
        candidates = {
            "chrome": "Google Chrome",
            "telegram": "Telegram",
            "spotify": "Spotify",
            "youtube": "YouTube",
            "discord": "Discord",
        }

        # exact
        res = fuzzy_resolve("chrome", candidates)
        self.assertIsNotNone(res)
        self.assertEqual(res.target_value, "Google Chrome")

        # typo / inflection
        res = fuzzy_resolve("телеграмму", candidates)
        # Note: "телеграмму" has length 10, "telegram" is in latin, so Russian candidated in dict:
        candidates_ru = {
            "хром": "Google Chrome",
            "телеграм": "Telegram",
            "спотифай": "Spotify",
            "споти": "Spotify",
            "ютуб": "YouTube",
            "ютубчик": "YouTube",
            "дискорд": "Discord",
        }
        res_ru = fuzzy_resolve("телеграмму", candidates_ru)
        self.assertIsNotNone(res_ru)
        self.assertEqual(res_ru.target_value, "Telegram")

        res_sp = fuzzy_resolve("спотик", candidates_ru)
        self.assertIsNotNone(res_sp)
        self.assertEqual(res_sp.target_value, "Spotify")

        res_yt = fuzzy_resolve("ютубчик", candidates_ru)
        self.assertIsNotNone(res_yt)
        self.assertEqual(res_yt.target_value, "YouTube")

    def test_fuzzy_resolve_no_match(self) -> None:
        candidates = ["chrome", "telegram", "spotify"]
        res = fuzzy_resolve("совершенно другое слово", candidates)
        self.assertIsNone(res)


if __name__ == "__main__":
    unittest.main()
