"""Shell capability — gives the agent operating-system access via `sh(...)`.

NOT sandboxed: commands run on the host with the engine's privileges. This is the
explicit power/trust tradeoff of the in-process backend; pair it with a container
execution backend for untrusted work.
"""

from __future__ import annotations

import subprocess
from typing import Any

from ..capability import BaseCapability, CapabilityContext

_SURFACE = """  sh(command, timeout=120) -> str
                Run a shell command on the operating system and return its
                combined stdout+stderr. Use it for anything the OS can do:
                inspect files, run tools, install packages, execute programs.
                Long output is returned in full — capture it in a variable and
                distill with llm() rather than printing it wholesale."""


class ShellCapability(BaseCapability):
    name = "shell"

    def __init__(self, *, default_timeout: int = 120, cwd: str | None = None):
        self.default_timeout = default_timeout
        self.cwd = cwd

    def bind(self, ctx: CapabilityContext) -> dict[str, Any]:
        cwd = self.cwd or ctx.workspace

        def sh(command: str, timeout: int | None = None) -> str:
            try:
                proc = subprocess.run(
                    command,
                    shell=True,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=timeout or self.default_timeout,
                )
            except subprocess.TimeoutExpired:
                return f"[command timed out after {timeout or self.default_timeout}s]"
            out = proc.stdout or ""
            if proc.stderr:
                out += ("\n" if out else "") + proc.stderr
            if proc.returncode != 0:
                out += f"\n[exit code {proc.returncode}]"
            return out.strip() or "(no output)"

        return {"sh": sh}

    def surface(self) -> str:
        return _SURFACE
