"""The `Capability` seam — what an Agent can touch.

vomero mounted a single `Source` (a data corpus). Cordyceps generalizes that to a
*set* of capabilities, each contributing two things to a run:

  1. **REPL names** (`bind`) — callables/objects injected into the execution
     namespace. Because the agent's only action is running code, a capability is
     just functions the model can call from Python: `sh("ls")`, `read_file(p)`,
     `web_search(q)`. This is the "tools as code" model — strictly more
     composable than a flat JSON tool menu, since the model can loop over and
     combine results in one step.

  2. **A prompt fragment** (`surface`) — the lines spliced into the system prompt
     that tell the model those names exist and how to use them.

A RAG corpus, a shell, a filesystem, an MCP tool registry — all are capabilities.
New ones slot in by implementing these two members; the engine stays agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class CapabilityContext:
    """Run context handed to `bind`, so a capability can wire itself to the live
    run (its depth in the recursion, a shared workspace directory, …)."""

    depth: int = 0
    workspace: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Capability(Protocol):
    """Something an Agent can use, exposed as REPL names + a prompt fragment."""

    name: str

    def bind(self, ctx: CapabilityContext) -> dict[str, Any]:
        """Return the REPL names this capability injects (name -> object)."""
        ...

    def surface(self) -> str:
        """The system-prompt block describing this capability's REPL names."""
        ...

    def subset(self, selector: Any) -> "Capability":
        """A capability scoped for a recursive sub-agent. Default is identity —
        only data-source capabilities (a corpus narrowed to some docs) override
        this; shell/fs/tools pass straight through."""
        ...


class BaseCapability:
    """Convenience base: gives `subset` an identity default so most capabilities
    only implement `bind` and `surface`."""

    name: str = "capability"

    def subset(self, selector: Any) -> "BaseCapability":
        return self
