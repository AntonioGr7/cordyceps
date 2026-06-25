"""Runtime configuration, read from the environment (and a local .env).

Cordyceps defaults to Claude via the native Anthropic client. The OpenAI-compat
client is kept for any OpenAI-compatible endpoint (vLLM, OpenRouter, …).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:  # optional, convenient for local dev
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


@dataclass
class Settings:
    """All knobs Cordyceps reads from the environment.

    `provider` selects the LLM backend: "anthropic" (default, native) or
    "openai" (any OpenAI-compatible server via `base_url`).
    """

    provider: str = "anthropic"
    model: str = "claude-opus-4-8"
    base_url: str | None = None
    api_key: str | None = None
    max_tokens: int = 16_000
    effort: str | None = None  # low|medium|high|xhigh|max (Anthropic only)

    # Agent loop limits
    max_steps: int = 24
    max_depth: int = 3
    # Hard cap (chars) on a single tool result before it enters the transcript.
    max_output_chars: int = 10_000
    # Fan-out width for llm_batched(...) — max concurrent flat sub-calls.
    max_parallel_calls: int = 8
    # Global budget across the WHOLE run tree (root + recursion). 0 = unlimited.
    max_total_tokens: int = 0
    max_total_calls: int = 0

    # Where the model's REPL code runs: "inprocess" (fast, unsandboxed) or
    # "gvisor" (each step in a gVisor container). Defaults to inprocess.
    exec_backend: str = "inprocess"

    # Context / compaction. When live context crosses compact_ratio * window,
    # the middle of the transcript is summarized. compact_ratio <= 0 disables.
    context_window: int = 200_000
    compact_ratio: float = 0.8
    compact_keep_recent: int = 6
    compact_min_reclaim: int = 2048

    # Live plan/TODO checklist driven by the model.
    enable_planning: bool = False
    planning_root_only: bool = False
    # Let the model ask the user for help when stuck.
    enable_interaction: bool = True
    interaction_root_only: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            provider=os.getenv("CORDYCEPS_PROVIDER", "anthropic"),
            model=os.getenv("CORDYCEPS_MODEL", "claude-opus-4-8"),
            base_url=(
                os.getenv("CORDYCEPS_BASE_URL")
                or os.getenv("ANTHROPIC_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
            ),
            api_key=(
                os.getenv("CORDYCEPS_API_KEY")
                or os.getenv("ANTHROPIC_API_KEY")
                or os.getenv("OPENAI_API_KEY")
            ),
            max_tokens=int(os.getenv("CORDYCEPS_MAX_TOKENS", "16000")),
            effort=os.getenv("CORDYCEPS_EFFORT") or None,
            max_steps=int(os.getenv("CORDYCEPS_MAX_STEPS", "24")),
            max_depth=int(os.getenv("CORDYCEPS_MAX_DEPTH", "3")),
            max_output_chars=int(os.getenv("CORDYCEPS_MAX_OUTPUT_CHARS", "10000")),
            max_parallel_calls=int(os.getenv("CORDYCEPS_MAX_PARALLEL_CALLS", "8")),
            max_total_tokens=int(os.getenv("CORDYCEPS_MAX_TOTAL_TOKENS", "0")),
            max_total_calls=int(os.getenv("CORDYCEPS_MAX_TOTAL_CALLS", "0")),
            exec_backend=os.getenv("CORDYCEPS_EXEC_BACKEND", "inprocess"),
            context_window=int(os.getenv("CORDYCEPS_CONTEXT_WINDOW", "200000")),
            compact_ratio=float(os.getenv("CORDYCEPS_COMPACT_RATIO", "0.8")),
            compact_keep_recent=int(os.getenv("CORDYCEPS_COMPACT_KEEP_RECENT", "6")),
            compact_min_reclaim=int(os.getenv("CORDYCEPS_COMPACT_MIN_RECLAIM", "2048")),
            enable_planning=os.getenv("CORDYCEPS_PLAN", "").lower() in ("1", "true", "yes"),
            planning_root_only=os.getenv("CORDYCEPS_PLAN_ROOT_ONLY", "").lower()
            in ("1", "true", "yes"),
            enable_interaction=os.getenv("CORDYCEPS_INTERACTIVE", "true").lower()
            in ("1", "true", "yes"),
            interaction_root_only=os.getenv("CORDYCEPS_ASK_ROOT_ONLY", "").lower()
            in ("1", "true", "yes"),
        )
