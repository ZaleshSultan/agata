"""
Tests for TTS phrase caching and short confirmation phrases.
"""

import pytest
from tts.xtts_engine import _PHRASE_CACHE, clear_phrase_cache, get_phrase_cache_size


def test_phrase_cache_operations():
    clear_phrase_cache()
    assert get_phrase_cache_size() == 0

    # Store a dummy audio entry
    dummy_wav = b"RIFF....WAVEfmt ...."
    _PHRASE_CACHE["закрыть"] = dummy_wav

    assert get_phrase_cache_size() == 1
    assert _PHRASE_CACHE.get("закрыть") == dummy_wav

    clear_phrase_cache()
    assert get_phrase_cache_size() == 0
