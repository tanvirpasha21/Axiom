"""
LLM Backend abstraction for AXIOM.
Supports cloud (Anthropic, OpenRouter) and local (Ollama) LLM providers.

Usage:
    # Use Anthropic Claude (default)
    agent = LiteratureAgent()

    # Use OpenRouter (any model via openrouter.ai)
    agent = LiteratureAgent(backend="openrouter", model="openai/gpt-4o")
    agent = LiteratureAgent(backend="openrouter", model="anthropic/claude-3.5-sonnet")
    agent = LiteratureAgent(backend="openrouter", model="google/gemini-pro-1.5")

    # Use local Llama via Ollama
    agent = LiteratureAgent(backend="ollama", model="llama2")
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import json
from typing import Optional
import anthropic
import httpx


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def query(self, system: str, prompt: str, max_tokens: int = 2048) -> str:
        pass

    @abstractmethod
    def close(self):
        pass


class AnthropicBackend(LLMBackend):
    """Claude API backend via Anthropic."""

    def __init__(self, api_key: str, model: str = "claude-opus-4-5"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def query(self, system: str, prompt: str, max_tokens: int = 2048) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def close(self):
        pass


class OpenRouterBackend(LLMBackend):
    """
    OpenRouter backend — access 200+ models through a single OpenAI-compatible API.

    Popular models:
        anthropic/claude-3.5-sonnet
        anthropic/claude-opus-4-5
        openai/gpt-4o
        openai/gpt-4o-mini
        google/gemini-pro-1.5
        meta-llama/llama-3.1-70b-instruct
        mistralai/mixtral-8x7b-instruct
        deepseek/deepseek-r1

    Full model list: https://openrouter.ai/models

    Usage:
        # Via environment variable (recommended)
        export OPENROUTER_API_KEY="sk-or-..."
        agent = LiteratureAgent(backend="openrouter", model="openai/gpt-4o")

        # Pass key directly
        agent = LiteratureAgent(
            backend="openrouter",
            api_key="sk-or-...",
            model="anthropic/claude-3.5-sonnet"
        )

        # Via CLI
        axiom search "fraud detection" --backend openrouter --model openai/gpt-4o
        OPENROUTER_API_KEY=sk-or-... axiom search "fraud detection" --backend openrouter
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "openai/gpt-4o-mini"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        site_url: Optional[str] = None,
        site_name: Optional[str] = "AXIOM Literature Agent",
    ):
        """
        Args:
            api_key: OpenRouter API key (starts with sk-or-...)
                     Get yours at https://openrouter.ai/keys
            model: Model identifier in provider/name format (e.g. "openai/gpt-4o")
                   See https://openrouter.ai/models for full list
            site_url: Optional — your app URL (shown in OpenRouter dashboard)
            site_name: Optional — your app name (shown in OpenRouter dashboard)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai"
            )

        extra_headers = {}
        if site_url:
            extra_headers["HTTP-Referer"] = site_url
        if site_name:
            extra_headers["X-Title"] = site_name

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.OPENROUTER_BASE_URL,
            default_headers=extra_headers or None,
        )
        self.model = model

    def query(self, system: str, prompt: str, max_tokens: int = 2048) -> str:
        """Query a model via OpenRouter."""
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    def close(self):
        pass


class OllamaBackend(LLMBackend):
    """Local LLM backend via Ollama."""

    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.client = httpx.Client(base_url=base_url, timeout=300.0)
        try:
            resp = self.client.get("/api/tags")
            if resp.status_code != 200:
                raise ConnectionError(
                    f"Failed to connect to Ollama at {base_url}. "
                    f"Make sure Ollama is running: ollama serve"
                )
        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {base_url}.\n"
                f"Please start Ollama: ollama serve\n"
                f"Or install from: https://ollama.ai"
            )

    def query(self, system: str, prompt: str, max_tokens: int = 2048) -> str:
        full_prompt = f"{system}\n\n{prompt}"
        resp = self.client.post(
            "/api/generate",
            json={"model": self.model, "prompt": full_prompt, "stream": False, "num_predict": max_tokens},
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama query failed: {resp.status_code}\n{resp.text}\n"
                f"Make sure model '{self.model}' is pulled: ollama pull {self.model}"
            )
        return resp.json().get("response", "").strip()

    def close(self):
        self.client.close()


class OpenAIBackend(LLMBackend):
    """OpenAI API backend."""

    def __init__(self, api_key: str, model: str = "gpt-4", base_url: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.model = model

    def query(self, system: str, prompt: str, max_tokens: int = 2048) -> str:
        message = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return message.choices[0].message.content

    def close(self):
        pass


def get_backend(
    backend_type: str = "anthropic",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> LLMBackend:
    """
    Factory function to instantiate an LLM backend.

    Args:
        backend_type: "anthropic", "openrouter", "ollama", or "openai"
        api_key: API key (required for anthropic/openrouter/openai)
        model: Model name
        base_url: Base URL for openai-compatible APIs

    Returns:
        LLMBackend instance
    """
    import os

    backend_type = backend_type.lower().strip()

    if backend_type == "anthropic":
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No Anthropic API key found.\n"
                "Either pass api_key= or set ANTHROPIC_API_KEY in your environment.\n"
                "Get a key at: https://console.anthropic.com"
            )
        return AnthropicBackend(api_key=resolved_key, model=model or "claude-opus-4-5")

    elif backend_type == "openrouter":
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No OpenRouter API key found.\n"
                "Either pass api_key= or set OPENROUTER_API_KEY in your environment.\n"
                "Get a free key at: https://openrouter.ai/keys"
            )
        return OpenRouterBackend(
            api_key=resolved_key,
            model=model or OpenRouterBackend.DEFAULT_MODEL,
        )

    elif backend_type == "ollama":
        return OllamaBackend(
            model=model or "llama2",
            base_url=base_url or "http://localhost:11434",
        )

    elif backend_type == "openai":
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No OpenAI API key found.\n"
                "Either pass api_key= or set OPENAI_API_KEY in your environment."
            )
        return OpenAIBackend(api_key=resolved_key, model=model or "gpt-4", base_url=base_url)

    else:
        raise ValueError(
            f"Unknown backend: {backend_type}\n"
            f"Supported: anthropic, openrouter, ollama, openai"
        )
