"""Cordyceps — a recursive, code-action agent and decomposition substrate.

The base unit is `Agent`: a described, capable agent whose only action is running
code in a persistent REPL, with capabilities (shell, filesystem, tools, data)
exposed as REPL functions and `spawn()` for recursive decomposition.
"""

from __future__ import annotations

from .agent import Agent, build_engine
from .capability import BaseCapability, Capability, CapabilityContext
from .config import Settings
from .engine import AgentEngine, RunResult, Step
from .observability import RunLogger
from .orchestrator import Decomposer, Decomposition, SubProblem, Swarm, SwarmResult

__all__ = [
    "Agent",
    "build_engine",
    "AgentEngine",
    "RunResult",
    "Step",
    "Settings",
    "Capability",
    "BaseCapability",
    "CapabilityContext",
    "RunLogger",
    "Swarm",
    "SwarmResult",
    "Decomposer",
    "Decomposition",
    "SubProblem",
]
