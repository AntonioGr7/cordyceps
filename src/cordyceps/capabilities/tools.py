"""Tool registry capability — arbitrary tools exposed as REPL callables.

This is the general "tools as code" surface: register any Python callable (a web
search, a database query, an API client, an MCP-backed function) and it becomes a
name the agent can call from its REPL — `result = web_search("...")`. Because the
action is code, the agent composes tools with loops and conditionals in a single
step instead of one round-trip per call.

The prompt surface is generated from each tool's signature and docstring, so a
well-documented function needs no extra wiring. MCP tools slot in by wrapping each
remote tool as a callable and registering it here.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

from ..capability import BaseCapability, CapabilityContext


@dataclass
class Tool:
    name: str
    func: Callable[..., Any]
    description: str = ""

    def signature(self) -> str:
        try:
            sig = str(inspect.signature(self.func))
        except (TypeError, ValueError):
            sig = "(...)"
        return f"{self.name}{sig}"

    def doc(self) -> str:
        if self.description:
            return self.description
        return inspect.getdoc(self.func) or ""


class ToolRegistry(BaseCapability):
    name = "tools"

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self._tools[t.name] = t

    def register(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str = "",
    ) -> "ToolRegistry":
        tool_name = name or func.__name__
        self._tools[tool_name] = Tool(name=tool_name, func=func, description=description)
        return self

    def bind(self, ctx: CapabilityContext) -> dict[str, Any]:
        return {name: tool.func for name, tool in self._tools.items()}

    def surface(self) -> str:
        if not self._tools:
            return ""
        lines = ["  Tools available as REPL functions:"]
        for tool in self._tools.values():
            doc = tool.doc().strip().splitlines()
            summary = doc[0] if doc else ""
            lines.append(f"    {tool.signature()}")
            if summary:
                lines.append(f"        {summary}")
        return "\n".join(lines)
