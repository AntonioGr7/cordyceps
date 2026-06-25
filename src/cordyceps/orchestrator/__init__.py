"""Layer 2 — the recursive problem-decomposition orchestrator.

`Swarm` decomposes a problem into a lazy tree-of-DAGs of `Agent`s built on the
Layer-1 engine. `Decomposer` is the splitting agent it drives.
"""

from __future__ import annotations

from .decomposer import Decomposer, Decomposition, SubProblem
from .swarm import Swarm, SwarmResult

__all__ = [
    "Swarm",
    "SwarmResult",
    "Decomposer",
    "Decomposition",
    "SubProblem",
]
