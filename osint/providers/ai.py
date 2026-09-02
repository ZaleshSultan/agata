"""Minimal optional providers for image clues and report assistance only."""

from __future__ import annotations

import base64

import httpx

import config
from osint.providers.base import AIAnalysis


class ProviderUnavailableError(RuntimeError): pass

class GeminiProvider:
    name = "gemini"
    def __init__(self, model: str, api_key: str) -> None: self.model, self.api_key = model, api_key
    async def analyze_image(self, image: bytes, prompt: str) -> AIAnalysis:
        if not self.api_key: raise ProviderUnavailableError("GEMINI_API_KEY is not configured.")
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(image).decode()}}]}]}
        async with httpx.AsyncClient(timeout=config.OSINT_REQUEST_TIMEOUT_SECONDS) as client: response = await client.post(endpoint, json=payload)
        response.raise_for_status(); data = response.json(); return AIAnalysis(data["candidates"][0]["content"]["parts"][0].get("text", ""))

class OpenRouterProvider:
    name = "openrouter"
    def __init__(self, model: str, api_key: str) -> None: self.model, self.api_key = model, api_key
    async def analyze_image(self, image: bytes, prompt: str) -> AIAnalysis:
        if not self.api_key: raise ProviderUnavailableError("OPENROUTER_API_KEY is not configured.")
        data_url = "data:image/png;base64," + base64.b64encode(image).decode()
        payload = {"model": self.model, "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": data_url}}]}]}
        async with httpx.AsyncClient(timeout=config.OSINT_REQUEST_TIMEOUT_SECONDS) as client: response = await client.post("https://openrouter.ai/api/v1/chat/completions", headers={"Authorization": f"Bearer {self.api_key}"}, json=payload)
        response.raise_for_status(); return AIAnalysis(response.json()["choices"][0]["message"]["content"])

class LocalOpenAIProvider(OpenRouterProvider):
    """OpenAI-compatible local endpoint, used only when explicitly configured."""
    name = "local"
    def __init__(self, model: str, base_url: str, api_key: str = "") -> None:
        super().__init__(model, api_key); self.base_url = base_url.rstrip("/")
    async def analyze_image(self, image: bytes, prompt: str) -> AIAnalysis:
        if not self.base_url: raise ProviderUnavailableError("OSINT_AI_BASE_URL is not configured.")
        data_url = "data:image/png;base64," + base64.b64encode(image).decode()
        payload = {"model": self.model, "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": data_url}}]}]}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        async with httpx.AsyncClient(timeout=config.OSINT_REQUEST_TIMEOUT_SECONDS) as client: response = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
        response.raise_for_status(); return AIAnalysis(response.json()["choices"][0]["message"]["content"])

def configured_providers() -> list[GeminiProvider | OpenRouterProvider | LocalOpenAIProvider]:
    providers = []
    settings = ((config.OSINT_AI_PROVIDER.casefold(), config.OSINT_AI_MODEL, config.OSINT_AI_API_KEY), (config.OSINT_AI_FALLBACK_PROVIDER.casefold(), config.OSINT_AI_FALLBACK_MODEL, config.OSINT_AI_FALLBACK_API_KEY))
    for provider, model, key in settings:
        if provider == "gemini" and model: providers.append(GeminiProvider(model, key or config.GEMINI_API_KEY))
        elif provider == "openrouter" and model: providers.append(OpenRouterProvider(model, key or config.OPENROUTER_API_KEY))
        elif provider == "local" and model: providers.append(LocalOpenAIProvider(model, config.OSINT_AI_BASE_URL, key))
    return providers

async def analyze_with_fallback(image: bytes, prompt: str) -> tuple[AIAnalysis, str]:
    errors = []
    for provider in configured_providers():
        try: return await provider.analyze_image(image, prompt), provider.name
        except Exception as exc: errors.append(f"{provider.name}: {exc}")
    raise ProviderUnavailableError("No configured AI provider succeeded: " + "; ".join(errors or ["none configured"]))

def configured_provider() -> GeminiProvider | OpenRouterProvider | LocalOpenAIProvider | None:
    return next(iter(configured_providers()), None)
