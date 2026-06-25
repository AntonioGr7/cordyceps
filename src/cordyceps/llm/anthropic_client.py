"""Native Anthropic client.

Translates Cordyceps' wire-neutral Message/ToolSpec types to the Anthropic
Messages API and back. This is the SOTA default for the base Agent — it talks to
Claude directly (not through an OpenAI-compat shim), so it gets first-class tool
use, prompt caching, and the latest models.

Design notes:
  * Anthropic carries the system prompt OUT of the message list (a top-level
    `system` arg), so we split it out during translation.
  * Tool results live in a `user` turn as `tool_result` blocks; the engine emits
    one `tool` message per result, so we merge a contiguous run of them into a
    single user turn (the API pairs them with the preceding assistant tool_use).
  * We always stream and assemble the final message — `max_tokens` for a
    code-writing agent is large enough that a non-streaming call risks the SDK's
    ~10-minute HTTP timeout.
  * Extended thinking is OFF by default. Replaying thinking blocks across turns
    requires echoing them back verbatim (with signatures), which the wire-neutral
    Message can't yet carry; enabling it without that round-trip would break the
    next request. Depth is instead steered with `effort` (output_config), which
    needs no block replay. Thinking-block round-tripping is a planned extension.
"""

from __future__ import annotations

from typing import Any

import anthropic

from .base import LLMResponse, Message, ToolCall, ToolSpec, Usage

DEFAULT_MODEL = "claude-opus-4-8"


class AnthropicClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 16_000,
        effort: str | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        # output_config effort: "low" | "medium" | "high" | "xhigh" | "max".
        # None => provider default (high). Steers reasoning depth without the
        # thinking-block replay that adaptive thinking would require.
        self.effort = effort
        self._client = anthropic.Anthropic(api_key=api_key, base_url=base_url)

    # -- translation: ours -> Anthropic ---------------------------------
    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
        system_parts = [m.content for m in messages if m.role == "system" and m.content]
        rest = [m for m in messages if m.role != "system"]
        system = "\n\n".join(system_parts) if system_parts else None
        return system, rest

    @staticmethod
    def _to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        pending_tool_results: list[dict[str, Any]] = []

        def flush_tool_results() -> None:
            if pending_tool_results:
                out.append({"role": "user", "content": list(pending_tool_results)})
                pending_tool_results.clear()

        for m in messages:
            if m.role == "tool":
                # Merge a contiguous run of tool results into one user turn.
                pending_tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": m.content or "",
                    }
                )
                continue

            flush_tool_results()

            if m.role == "assistant" and m.tool_calls:
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": m.role, "content": m.content or ""})

        flush_tool_results()
        return out

    @staticmethod
    def _to_anthropic_tools(tools: list[ToolSpec] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in tools
        ]

    # -- the protocol method --------------------------------------------
    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float | None = None,  # accepted for protocol parity; ignored
    ) -> LLMResponse:
        system, rest = self._split_system(messages)
        kwargs: dict[str, Any] = {
            "model": model or self.model,
            "max_tokens": self.max_tokens,
            "messages": self._to_anthropic_messages(rest),
        }
        if system:
            kwargs["system"] = system
        anthropic_tools = self._to_anthropic_tools(tools)
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}

        # Stream + assemble: large max_tokens would risk the SDK's HTTP timeout
        # on a plain (non-streaming) call.
        with self._client.messages.stream(**kwargs) as stream:
            msg = stream.get_final_message()

        content_text: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in msg.content:
            if block.type == "text":
                content_text.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        usage = None
        if msg.usage is not None:
            prompt = (
                (msg.usage.input_tokens or 0)
                + (getattr(msg.usage, "cache_creation_input_tokens", 0) or 0)
                + (getattr(msg.usage, "cache_read_input_tokens", 0) or 0)
            )
            usage = Usage(prompt_tokens=prompt, completion_tokens=msg.usage.output_tokens or 0)

        return LLMResponse(
            content="".join(content_text) or None,
            tool_calls=tool_calls,
            usage=usage,
        )
