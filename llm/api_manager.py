"""
File: api_manager.py

Description:
This module defines the APIManager class, which serves as the unified
interface between the simulation backend and all LLM providers.

"""

"""
Use Case:
from backend.llm.api_manager import APIManager
api = APIManager()
prompt = ('Hello, who are you?')
response = api.generate(prompt)
"""



from time import perf_counter
from typing import Optional
from llm.providers import DeepSeekProvider, OpenAIProvider, OllamaProvider
from llm.routing import (
    ResolvedLLMRoute,
    rate_limit_fallback_route,
    resolve_llm_route,
)

debug = False
SUPPORTED_PROVIDER_NAMES = ("ollama", "openai", "deepseek")


class APIManager:
    """
    Unified LLM manager and router.
    """

    def __init__(self,
                 provider_name: str = "ollama",
                 task_name: str = ""):
        """
        Initialize APIManager.

        INPUT
          provider_name: "openai", "ollama", or "deepseek"
        OUTPUT
          None
        """

        self.requested_provider_name = provider_name.lower()
        self.task_name = str(task_name or "").strip()
        self.route: ResolvedLLMRoute = resolve_llm_route(
            task_name=self.task_name,
            requested_provider=self.requested_provider_name,
            requested_model=None,
        )
        self.provider_name = self.route.provider_name
        self.primary_route = self.route
        self.rate_limit_fallback_used = False
        self.rate_limit_error = ""
        self.call_attempts: list[dict] = []

        self.openai_provider = OpenAIProvider()
        self.ollama_provider = OllamaProvider()
        self.deepseek_provider = DeepSeekProvider()

        self.provider = self._select_provider()

    def _select_provider(self):
        """
        Select provider instance.

        INPUT
          None
        OUTPUT
          Provider instance.
        EXAMPLE OUTPUT
          <OpenAIProvider object>
        """

        if self.provider_name == "openai":
            return self.openai_provider

        if self.provider_name == "ollama":
            return self.ollama_provider

        if self.provider_name == "deepseek":
            return self.deepseek_provider

        raise ValueError(f"Unsupported provider: {self.provider_name}")

    def set_provider(self, provider_name: str):
        """
        Dynamically switch provider.

        INPUT
          provider_name: "openai", "ollama", or "deepseek"
        OUTPUT
          None
        """

        self.requested_provider_name = provider_name.lower()
        self.route = resolve_llm_route(
            task_name=self.task_name,
            requested_provider=self.requested_provider_name,
            requested_model=None,
        )
        self.provider_name = self.route.provider_name
        self.provider = self._select_provider()

        if debug:
            print("[APIManager] Switched provider to:", self.provider_name)

    def generate(self,
                 prompt: str,
                 persona=None,
                 model: Optional[str] = None,
                 temperature: Optional[float] = None) -> str:
        """
        Generate text from LLM.

        INPUT
          prompt: Input prompt string.
          persona: Optional persona for model override.
          model: Optional model override.
          temperature: Optional sampling temperature. Provider defaults remain
            unchanged when omitted.
        OUTPUT
          Generated text string.
        EXAMPLE OUTPUT
          "Sure, I will go to work at 9 AM."
        """

        route = resolve_llm_route(
            task_name=self.task_name,
            requested_provider=self.requested_provider_name,
            requested_model=model,
        )
        self.primary_route = route
        self.rate_limit_fallback_used = False
        self.rate_limit_error = ""
        self.call_attempts = []
        self._activate_route(route)
        model_name = route.model if route.overridden else model

        if persona is not None and not route.overridden:
            if hasattr(persona, "scratch"):
                if hasattr(persona.scratch, "model"):
                    model_name = persona.scratch.model

        if debug:
            print("[APIManager] Task:", self.task_name or "unregistered")
            print("[APIManager] Routing profile:", route.profile_name)
            print("[APIManager] Provider:", self.provider_name)
            print("[APIManager] Model:", model_name)
            print("[APIManager] Temperature:", temperature)

        try:
            return self._generate_once(
                prompt,
                model_name=model_name,
                temperature=temperature,
            )
        except Exception as error:
            fallback_route = (
                rate_limit_fallback_route(route)
                if _is_rate_limit_error(error)
                else None
            )
            if fallback_route is None:
                raise

            self.rate_limit_fallback_used = True
            self.rate_limit_error = str(error)
            self._activate_route(fallback_route)
            try:
                return self._generate_once(
                    prompt,
                    model_name=fallback_route.model,
                    temperature=temperature,
                )
            except Exception as fallback_error:
                raise RuntimeError(
                    "Hybrid LLM rate-limit fallback failed: "
                    f"{route.provider_name} was rate limited; "
                    f"{fallback_route.provider_name} then failed with {fallback_error}"
                ) from fallback_error

    def _activate_route(self, route: ResolvedLLMRoute) -> None:
        self.route = route
        self.provider_name = route.provider_name
        self.provider = self._select_provider()

    def _generate_once(
        self,
        prompt: str,
        *,
        model_name: Optional[str],
        temperature: Optional[float],
    ) -> str:
        started = perf_counter()
        try:
            output = self.provider.generate(
                prompt,
                model_name,
                temperature=temperature,
            )
        except Exception as error:
            self.call_attempts.append({
                "provider_name": self.provider_name,
                "model": model_name or "provider default",
                "duration_seconds": round(perf_counter() - started, 6),
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            })
            raise
        self.call_attempts.append({
            "provider_name": self.provider_name,
            "model": model_name or "provider default",
            "duration_seconds": round(perf_counter() - started, 6),
            "status": "success",
            "error_type": "",
            "error": "",
        })
        return output

    def embed(self,
              text: str,
              persona=None) -> list:
        """
        Generate embedding vector.

        INPUT
          text: Input text.
          persona: Optional persona.
        OUTPUT
          Embedding vector list.
        EXAMPLE OUTPUT
          [0.0123, -0.4421, ...]
        """

        if debug:
            print("[APIManager] Embedding request")

        return self.provider.embed(text)


def _is_rate_limit_error(error: Exception) -> bool:
    status_code = getattr(error, "code", None)
    response = getattr(error, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    if status_code == 429:
        return True
    message = str(error or "").lower()
    return any(
        marker in message
        for marker in (
            "httperror 429",
            "http 429",
            "429 client error",
            "rate_limit_exceeded",
            "rate limit reached",
            "too many requests",
        )
    )
