"""Run Cordyceps as a Terminal-Bench agent.

    pip install terminal-bench          # plus Docker running
    pip install -e .                    # cordyceps, from the repo root
    # configure .env for your provider (anthropic/openai), then from repo root:
    tb run --agent-import-path examples.terminal_bench.cordyceps_tb_agent:CordycepsAgent -n 2

Architecture: Cordyceps' reasoning loop runs on the host; the benchmark hands us a
`TmuxSession` into the task's container. We mount a `TmuxShellCapability` whose
`sh(...)` routes commands INTO that session (not the host) and parses their output
+ exit code back out — so the agent does its work inside the container exactly the
way the benchmark expects, while keeping Cordyceps' code-action + spawn machinery.

The tmux output parsing (the sentinel trick below) is the part most likely to need
tuning against the real harness; it is intentionally simple and well-marked.
"""

from __future__ import annotations

import itertools
import os
import re
from pathlib import Path

from terminal_bench.agents.base_agent import AgentResult, BaseAgent
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.terminal.tmux_session import TmuxSession

from cordyceps.capability import BaseCapability, CapabilityContext
from cordyceps.config import Settings
from cordyceps.agent import build_engine
from cordyceps.observability import RunLogger
from cordyceps.usage import UsageMeter

_counter = itertools.count()

_SURFACE = """  sh(command, timeout=60) -> str
                Run a shell command INSIDE the task's terminal (a real container)
                and return its combined output; the exit code is appended as a
                trailing line like '[exit code 0]'. The command must RETURN —
                start any long-running process (a server, daemon, watcher) in the
                background with a trailing ' &', or the terminal blocks. For slow
                commands (package installs, compiles) pass a bigger timeout, e.g.
                sh('pip install numpy scipy', timeout=300). If a call reports it
                timed out, the stuck process was interrupted for you; background
                it or shorten it and retry — do NOT fall back to Python file/OS
                operations, which run on the wrong machine."""


class TmuxShellCapability(BaseCapability):
    """A shell capability whose sh() executes in a Terminal-Bench tmux session.

    sh() never raises: a timeout (a non-returning command) is caught, the stuck
    process is interrupted with Ctrl-C to unwedge the terminal, and a clear
    message is returned so the model can recover instead of crashing the loop.
    """

    name = "terminal"

    def __init__(self, session: TmuxSession, default_timeout: float = 60.0):
        self.session = session
        self.default_timeout = default_timeout

    def bind(self, ctx: CapabilityContext) -> dict:
        def sh(command: str, timeout: float | None = None) -> str:
            n = next(_counter)
            start, end = f"__CDX_S_{n}__", f"__CDX_E_{n}__"
            # Bracket the command with markers printed on their OWN lines; the
            # end marker carries the exit code. The typed command line also
            # contains the marker substrings (it's echoed in the pane), so we
            # match markers only when they are a line by themselves / a line
            # prefix — never inside the long command-echo line.
            wrapped = f"echo {start}; {command}; echo {end}$?"
            tmo = timeout or self.default_timeout
            try:
                self.session.send_keys(wrapped + "\n", block=True, max_timeout_sec=tmo)
            except TimeoutError:
                self._interrupt()
                return (
                    f"[command did not return within {tmo:.0f}s and was "
                    "interrupted (Ctrl-C). If it is a server or other long-running "
                    "process, start it in the background with a trailing ' &'. If "
                    "it is just slow, retry with a larger timeout=.]"
                )
            except Exception as e:  # never let the loop crash on a shell hiccup
                return f"[sh error: {type(e).__name__}: {e}]"
            screen = self.session.capture_pane(capture_entire=True)
            return _extract(screen, start, end)

        return {"sh": sh}

    def _interrupt(self) -> None:
        """Send Ctrl-C to free a wedged terminal; ignore any failure."""
        try:
            self.session.send_keys("C-c", block=False, min_timeout_sec=0.5)
        except Exception:
            pass

    def surface(self) -> str:
        return _SURFACE


def _extract(screen: str, start: str, end: str) -> str:
    lines = screen.splitlines()
    # last line that IS the start marker (the printed output, not the echoed cmd)
    s_idx = max((i for i, ln in enumerate(lines) if ln.strip() == start), default=None)
    e_idx = e_code = None
    for i, ln in enumerate(lines):
        m = re.match(rf"^{re.escape(end)}(\d+)\s*$", ln.strip())
        if m and (s_idx is None or i > s_idx):
            e_idx, e_code = i, int(m.group(1))
    if s_idx is None or e_idx is None:
        return screen.strip() or "(no output captured)"
    body = "\n".join(lines[s_idx + 1 : e_idx]).strip()
    return (body or "(no output)") + f"\n[exit code {e_code}]"


_ROLE = (
    "You operate a Linux terminal to accomplish a task.\n\n"
    "CRITICAL — where your actions take effect: your Python REPL runs on a "
    "SEPARATE control machine, NOT in the task environment. Python file, OS, and "
    "network operations (open(), pathlib, os, subprocess, requests, …) act on the "
    "control machine and have ZERO effect on the task — the grader cannot see "
    "them. The ONE bridge to the task environment is sh(command): it runs that "
    "command in the actual task terminal (a container) and returns its output.\n\n"
    "Therefore: do EVERYTHING that must affect the task through sh(). To create a "
    "file, use sh(\"cat > file <<'EOF'\\n...\\nEOF\") or sh(\"printf ... > file\") "
    "— never Python's open()/write_text(). Use the Python REPL only to build "
    "command strings, parse sh() output, and control your loop.\n\n"
    "Work iteratively: run a command, read its output, run the next. Verify with "
    "sh() (re-read the file, check exit codes) before finishing. Make the real "
    "changes in the task terminal, then call answer(...) with a short summary.\n\n"
    "Rules that keep the terminal healthy:\n"
    "- Every command must RETURN. Start servers/daemons/long watchers in the "
    "background: sh('python3 server.py >/tmp/s.log 2>&1 &'), then sh('sleep 1; "
    "cat /tmp/s.log') to confirm. Never run a foreground process that doesn't "
    "exit — it freezes the terminal.\n"
    "- To run a multi-line script in the container, WRITE IT TO A FILE first, "
    "then run it: sh(\"cat > /tmp/run.py <<'EOF'\\n<your code>\\nEOF\") then "
    "sh('python3 /tmp/run.py'). Avoid inline 'python3 - <<PY' heredocs inside "
    "sh(...) — the nested quoting is error-prone.\n"
    "- Don't run interactive programs (vim, less, top, a bare python REPL). Use "
    "non-interactive equivalents.\n"
    "- If sh() reports a timeout, the process was already interrupted; background "
    "it or raise the timeout — never switch to Python open()/os to do the task."
)


class CordycepsAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "cordyceps"

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        settings = Settings.from_env()
        engine = build_engine(settings)
        meter = UsageMeter(
            max_total_tokens=settings.max_total_tokens,
            max_total_calls=settings.max_total_calls,
        )
        shell = TmuxShellCapability(session)

        # Log the full cordyceps trajectory next to the harness's own logs, so a
        # failed task can be debugged (every sh() call, output, and sub-answer).
        # Always record JSONL; stream the live split to stderr when
        # CORDYCEPS_TB_LIVE is set (handy while iterating).
        live = os.getenv("CORDYCEPS_TB_LIVE", "").lower() in ("1", "true", "yes")
        log = None
        if logging_dir is not None or live:
            jsonl = str(logging_dir / "cordyceps_run.jsonl") if logging_dir else None
            log = RunLogger(jsonl_path=jsonl, live=live, verbose=live)

        failure = FailureMode.NONE
        try:
            engine.run(instruction, [shell], instructions=_ROLE, meter=meter,
                       on_event=log)
        except Exception:
            failure = getattr(FailureMode, "UNKNOWN_AGENT_ERROR", FailureMode.NONE)
        finally:
            if log is not None:
                log.close()

        return AgentResult(
            total_input_tokens=meter.prompt_tokens,
            total_output_tokens=meter.completion_tokens,
            failure_mode=failure,
            timestamped_markers=[],
        )
