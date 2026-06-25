"""Agent — the SOTA standalone unit of Cordyceps.

An Agent is a *role contract* over the code-action engine: a name, a description
of the sub-task it solves, the capabilities it may use, and a budget. Point it at
a task and it reasons, writes and runs code, uses its tools and the OS, decomposes
via `spawn`, and returns an answer.

This is deliberately self-contained: lift `Agent` out of Cordyceps and it is a
complete, capable agent on its own. The Cordyceps orchestrator (the recursive
problem-decomposer) is a thin layer built on top of many of these.

    from cordyceps import Agent
    from cordyceps.capabilities import ShellCapability, FileSystemCapability

    agent = Agent(
        role="builder",
        description="You implement small, well-scoped programming tasks.",
        capabilities=[ShellCapability(), FileSystemCapability(root="./work")],
    )
    print(agent.solve("Create hello.py that prints 'hi' and run it."))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .capability import Capability
from .config import Settings
from .engine import AgentEngine, Compactor, RunResult, Step
from .execution import build_env_factory
from .llm import build_client
from .usage import UsageMeter


@dataclass
class Agent:
    """A described, capable agent over the code-action engine.

    `role` + `description` become the agent's identity/task contract in its system
    prompt — this is what lets it be composed as a specialized sub-agent.
    `max_total_tokens` / `max_total_calls` bound the whole run tree (0 = use the
    engine/settings default).
    """

    role: str
    description: str
    capabilities: list[Capability] = field(default_factory=list)
    settings: Settings | None = None
    engine: AgentEngine | None = None
    max_total_tokens: int = 0
    max_total_calls: int = 0

    def __post_init__(self) -> None:
        if self.engine is None:
            self.engine = build_engine(self.settings or Settings.from_env())

    def _instructions(self) -> str:
        header = f"You are '{self.role}'." if self.role else ""
        return (header + "\n" + self.description).strip()

    def solve(
        self,
        task: str,
        *,
        on_event: Callable[[Step], None] | None = None,
        ask_handler: Callable[[str], str] | None = None,
        return_trajectory: bool = False,
    ) -> str | RunResult:
        """Run the agent on `task` and return its answer (or a RunResult)."""
        meter = UsageMeter(
            max_total_tokens=self.max_total_tokens,
            max_total_calls=self.max_total_calls,
        )
        return self.engine.run(
            task,
            self.capabilities,
            instructions=self._instructions(),
            meter=meter,
            on_event=on_event,
            ask_handler=ask_handler,
            return_trajectory=return_trajectory,
        )


def build_engine(settings: Settings) -> AgentEngine:
    """Construct an AgentEngine from Settings — client, exec backend, compaction,
    and loop limits all wired from one config object."""
    client = build_client(settings)
    compactor = (
        Compactor(
            context_window=settings.context_window,
            ratio=settings.compact_ratio,
            keep_recent_messages=settings.compact_keep_recent,
            min_reclaim_tokens=settings.compact_min_reclaim,
        )
        if settings.compact_ratio > 0
        else None
    )
    return AgentEngine(
        client,
        env_factory=build_env_factory(settings),
        model=settings.model,
        max_steps=settings.max_steps,
        max_depth=settings.max_depth,
        max_output_chars=settings.max_output_chars,
        max_parallel_calls=settings.max_parallel_calls,
        compactor=compactor,
        enable_planning=settings.enable_planning,
        planning_root_only=settings.planning_root_only,
        enable_interaction=settings.enable_interaction,
        interaction_root_only=settings.interaction_root_only,
    )
