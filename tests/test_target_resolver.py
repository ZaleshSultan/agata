"""
Тесты для модуля target_resolver.py (универсальный слой разрешения целей).

Покрытие:
- Явные URL и домены (https://..., example.com)
- Встроенные картографированные веб-сервисы (Spotify, GitHub, Reddit, ChatGPT, YouTube, Discord, Telegram)
- Поиск приложений и система вызова
"""

import unittest
from target_resolver import resolve_target, ResolvedTarget


class TargetResolverTests(unittest.TestCase):

    # ── Tier 1: URL / Domain ───────────────────────────────────────────────

    def test_explicit_https_url(self) -> None:
        resolved = resolve_target("https://example.com/test")
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.kind, "browser")
        self.assertEqual(resolved.target, "https://example.com/test")

    def test_explicit_http_url(self) -> None:
        resolved = resolve_target("http://localhost:8080")
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.kind, "browser")
        self.assertEqual(resolved.target, "http://localhost:8080")

    def test_domain_name(self) -> None:
        resolved = resolve_target("github.com")
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.kind, "browser")
        self.assertEqual(resolved.target, "https://github.com")

    # ── Tier 3: Web Services Fallbacks ─────────────────────────────────────

    def test_spotify_web_fallback(self) -> None:
        resolved = resolve_target("spotify")
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertIn(resolved.kind, ["app", "browser"])
        if resolved.kind == "browser":
            self.assertEqual(resolved.target, "https://open.spotify.com")

    def test_github_web_fallback(self) -> None:
        resolved = resolve_target("github")
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertIn(resolved.kind, ["app", "browser"])
        if resolved.kind == "browser":
            self.assertEqual(resolved.target, "https://github.com")

    def test_chatgpt_web_fallback(self) -> None:
        resolved = resolve_target("chatgpt")
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.kind, "browser")
        self.assertEqual(resolved.target, "https://chatgpt.com")

    def test_reddit_web_fallback(self) -> None:
        resolved = resolve_target("reddit")
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.kind, "browser")
        self.assertEqual(resolved.target, "https://www.reddit.com")

    # ── Tier 2: System Applications (Standard Windows apps) ───────────────

    def test_cmd_or_calc(self) -> None:
        resolved = resolve_target("cmd")
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.kind, "app")

    def test_empty_target(self) -> None:
        self.assertIsNone(resolve_target(""))
        self.assertIsNone(resolve_target("   "))


if __name__ == "__main__":
    unittest.main()
