import asyncio
import pytest

from osint.providers.ai import GeminiProvider, LocalOpenAIProvider, ProviderUnavailableError


def test_missing_gemini_key_fails_cleanly() -> None:
    with pytest.raises(ProviderUnavailableError): asyncio.run(GeminiProvider("gemini-test", "").analyze_image(b"x", "describe"))


def test_missing_local_url_fails_cleanly() -> None:
    with pytest.raises(ProviderUnavailableError): asyncio.run(LocalOpenAIProvider("local", "").analyze_image(b"x", "describe"))
