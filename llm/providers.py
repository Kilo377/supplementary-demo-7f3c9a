from __future__ import annotations

"""
File: providers.py

Description:
This module defines LLM providers for the simulation backend.

It contains:

1. OpenAIProvider
2. OllamaProvider
3. DeepSeekProvider

The providers expose the unified generation interface:

- generate(prompts: str, model: str | None = None, temperature: float | None = None) -> str
- embed(text: str) -> list[float] where the backend supports embeddings

Ollama uses default model: qwen2.5:1.5b
DeepSeek is generation-only in this project because its chat API does not
provide the embedding endpoint expected by the retrieval modules.

This module does not perform routing logic.
Routing must be handled by APIManager.
"""

import os
import json
import urllib.request
import urllib.error
from typing import List, Optional

try:
    import requests  # type: ignore
except Exception:
    requests = None

debug = False
OPENAI_TIMEOUT_SECONDS = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "120"))


def _validated_temperature(value: float) -> float:
    temperature = float(value)
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature must be between 0 and 2.")
    return temperature


def _post_json(url: str, data: dict, *, headers: dict | None = None, timeout: int = 120) -> dict:
    if requests is not None:
        response = requests.post(url, headers=headers, json=data, timeout=timeout)
        response.raise_for_status()
        return response.json()

    body = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers or {"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        message = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTPError {error.code}: {message}") from error

class OllamaProvider:
    """
    Ollama local provider.
    Default model: qwen2.5:1.5b
    """

    def __init__(self,
                 base_url: str = "http://localhost:11434",
                 default_model: str = "qwen3.6:35b",
                 embedding_model: str = "bge-m3:latest"):
        """
        Initialize OllamaProvider.

        INPUT
          base_url: Ollama server base URL.
          default_model: Default generation model.
          embedding_model: Embedding model nam4132e.
        OUTPUT
          None
        """
        self.base_url = base_url
        self.default_model = default_model
        self.embedding_model = embedding_model

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate text using Ollama.

        INPUT
          prompt: Prompt string.
          model: Optional model override.
          temperature: Optional sampling temperature.
        OUTPUT
          Generated text string.
        EXAMPLE OUTPUT
          "Sure, let's start."
        """
        model_name = model or self.default_model

        data = {
            "model": model_name,
            "prompt": prompt,
            "think": False, #True, #False,
            "stream": False
        }
        if temperature is not None:
            data["options"] = {
                "temperature": _validated_temperature(temperature),
            }

        result = _post_json(
            f"{self.base_url}/api/generate",
            data,
            timeout=120,
        )

        return result.get("response", "")

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding using Ollama embedding endpoint.

        INPUT
          text: Input text.
        OUTPUT
          Embedding vector.
        EXAMPLE OUTPUT
          [0.0231, -0.1123, ...]
        """
        data = {
            "model": self.embedding_model,
            "prompt": text
        }

        result = _post_json(
            f"{self.base_url}/api/embeddings",
            data,
            timeout=60,
        )

        embedding = result.get("embedding", [])

        if debug:
            print("[Ollama] embedding length:", len(embedding))

        return embedding



class OpenAIProvider:
    """
    OpenAI API provider.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 default_model: str = "gpt-4.1",
                 embedding_model: str = "text-embedding-3-small"):
        """
        Initialize OpenAIProvider.

        INPUT
          api_key: OpenAI API key.
          default_model: Default chat model name.
          embedding_model: Embedding model name.
        OUTPUT
          None
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.default_model = default_model
        self.embedding_model = embedding_model
        self.base_url = "https://api.openai.com/v1"

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate text using OpenAI chat completion.

        INPUT
          prompts: Prompt string.
          model: Optional model override.
          temperature: Optional sampling temperature from 0 to 2.
        OUTPUT
          Generated text string.
        EXAMPLE OUTPUT
          "Hello, how can I help you today?"
        """
        model_name = model or self.default_model

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        if temperature is not None:
            data["temperature"] = _validated_temperature(temperature)

        result = _post_json(
            f"{self.base_url}/chat/completions",
            data,
            headers=headers,
            timeout=OPENAI_TIMEOUT_SECONDS,
        )

        if debug:
            print("[OpenAI] model:", model_name)

        return result["choices"][0]["message"]["content"]

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding using OpenAI embedding API.

        INPUT
          text: Input text.
        OUTPUT
          Embedding vector.
        EXAMPLE OUTPUT
          [0.0123, -0.4421, ...]
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.embedding_model,
            "input": text
        }

        result = _post_json(
            f"{self.base_url}/embeddings",
            data,
            headers=headers,
            timeout=OPENAI_TIMEOUT_SECONDS,
        )

        if debug:
            print("[OpenAI] embedding length:",
                  len(result["data"][0]["embedding"]))

        return result["data"][0]["embedding"]


class DeepSeekProvider:
    """DeepSeek's OpenAI-compatible chat-completions provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        thinking_enabled: Optional[bool] = None,
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = (
            base_url
            or os.getenv("DEEPSEEK_BASE_URL")
            or "https://api.deepseek.com"
        ).rstrip("/")
        self.default_model = (
            default_model
            or os.getenv("DEEPSEEK_MODEL")
            or "deepseek-v4-flash"
        )
        self.timeout_seconds = int(
            timeout_seconds
            if timeout_seconds is not None
            else os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "120")
        )
        self.thinking_enabled = (
            thinking_enabled
            if thinking_enabled is not None
            else os.getenv("DEEPSEEK_THINKING", "disabled").strip().lower()
            in {"1", "true", "yes", "on", "enabled"}
        )

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        if not self.api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Export it before using provider=deepseek."
            )
        data = {
            "model": model or self.default_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "thinking": {
                "type": "enabled" if self.thinking_enabled else "disabled"
            },
        }
        if temperature is not None:
            data["temperature"] = _validated_temperature(temperature)
        result = _post_json(
            f"{self.base_url}/chat/completions",
            data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout_seconds,
        )
        try:
            return str(result["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("DeepSeek response did not contain message content.") from error

    def embed(self, text: str) -> List[float]:
        raise NotImplementedError(
            "DeepSeek generation API does not provide the embedding endpoint used "
            "by this project. Use ollama or openai as the embedding provider."
        )
