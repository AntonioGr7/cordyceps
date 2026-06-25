"""Provider-agnostic LLM layer.

`base.py` defines the wire-neutral types and the `LLMClient` protocol; the engine
talks only to those. Each concrete provider translates to/from its own schema.
"""

from __future__ import annotations

from .base import LLMClient, LLMResponse, Message, ToolCall, ToolSpec, Usage

__all__ = [
    "LLMClient",
    "LLMResponse",
    "Message",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "build_client",
]


def build_client(settings) -> LLMClient:
    """Factory: map `settings.provider` to a concrete client."""
    provider = settings.provider.lower()
    if provider == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(
            model=settings.model,
            api_key=settings.api_key,
            base_url=settings.base_url,
            max_tokens=settings.max_tokens,
            effort=settings.effort,
        )
    if provider == "openai":
        from .openai_client import OpenAIClient

        return OpenAIClient(
            model=settings.model,
            base_url=settings.base_url,
            api_key=settings.api_key,
        )
    raise ValueError(
        f"Unknown provider {provider!r}. Implemented: 'anthropic', 'openai'. "
        "Add a client in cordyceps/llm/ that satisfies the LLMClient protocol."
    )
