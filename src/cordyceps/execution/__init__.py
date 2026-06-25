"""Execution environments — where the model's code actually runs.

The engine depends only on the `ExecutionEnvironment` ABC. The default backend is
in-process `exec` (fast, full-power, NOT sandboxed). A gVisor/container backend
implementing the same interface is the planned isolation drop-in.
"""

from __future__ import annotations

from typing import Any, Callable

from .base import ExecResult, ExecutionEnvironment
from .inprocess import InProcessEnvironment

__all__ = [
    "ExecResult",
    "ExecutionEnvironment",
    "InProcessEnvironment",
    "build_env_factory",
]


def build_env_factory(settings: Any) -> Callable[[], ExecutionEnvironment]:
    """Map `settings` to an `env_factory` for the engine.

    Returns `InProcessEnvironment` today. A `gvisor` backend slots in here behind
    the same `ExecutionEnvironment` interface without touching engine code.
    """
    backend = getattr(settings, "exec_backend", "inprocess")
    if backend != "inprocess":
        raise ValueError(
            f"exec_backend {backend!r} not available in this build; only "
            "'inprocess' is implemented so far."
        )
    return InProcessEnvironment
