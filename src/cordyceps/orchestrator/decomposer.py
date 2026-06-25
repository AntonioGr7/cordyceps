"""The Decomposer — an agent whose ONLY job is to split a problem.

Given a problem and a short description of the input currently available to it,
the decomposer returns one decision:

  * **ATOMIC** — the problem is small enough to solve directly (a worker agent
    will handle it), or
  * **SPLIT** — a small set of sub-problems plus their dependency edges (which
    sub-problem's output feeds which). Independent sub-problems can run in
    parallel; dependent ones form a pipeline.

This is deliberately a *structured* call, not free-form code: the decomposer
emits a plan, and the orchestrator (`Swarm`) executes it. Decomposition is lazy
and recursive — each sub-problem is itself re-decomposed only when it runs, with
its real resolved input in hand, so the tree is built in real time rather than
planned all at once.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..llm.base import LLMClient, Message
from ..usage import UsageMeter


@dataclass
class SubProblem:
    """One node of a decomposition: a scoped sub-problem and its dependencies."""

    id: str
    task: str
    role: str = ""  # specialized identity for the worker agent that solves it
    depends_on: list[str] = field(default_factory=list)  # ids whose output feeds this


@dataclass
class Decomposition:
    """The decomposer's decision for one problem."""

    atomic: bool
    subproblems: list[SubProblem] = field(default_factory=list)
    reasoning: str = ""


_SYSTEM = (
    "You are a problem-decomposition planner. Your ONLY job is to decide whether "
    "a problem should be solved as one unit or split into sub-problems — you do "
    "NOT solve it. Split a problem ONLY when it is genuinely complex and "
    "separable; prefer keeping it atomic when a single capable agent could do it "
    "in one focused effort. When you do split, produce the SMALLEST set of "
    "sub-problems that covers the work, and wire their dependencies precisely: a "
    "sub-problem depends on another when it needs that other's OUTPUT as input. "
    "Independent sub-problems will run in parallel; dependent ones form a "
    "pipeline. Never invent work that isn't required to solve the problem."
)

_INSTRUCTION = """\
Problem to assess:
{problem}

Input available to whoever solves this:
{input_summary}

Decide: is this ATOMIC (solve as one unit) or should it be SPLIT?

Respond with ONLY a JSON object, no prose, in exactly this shape:

{{
  "atomic": true | false,
  "reasoning": "one sentence on why",
  "subproblems": [
    {{
      "id": "a",
      "task": "self-contained statement of this sub-problem",
      "role": "short identity for the agent solving it, e.g. 'data cleaner'",
      "depends_on": []
    }},
    {{
      "id": "b",
      "task": "...",
      "role": "...",
      "depends_on": ["a"]
    }}
  ]
}}

If "atomic" is true, "subproblems" must be an empty list. Use short string ids
("a", "b", "c", ...). A sub-problem's "depends_on" lists the ids whose results it
needs as input; leave it empty for sub-problems that start from the original
input. Do NOT create dependency cycles."""


def _extract_json(text: str | None) -> dict | None:
    """Pull a JSON object out of a model reply, tolerating code fences / prose."""
    if not text:
        return None
    text = text.strip()
    # Strip ```json … ``` fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost {...} span.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


class Decomposer:
    """Wraps the planning LLM call + parsing into a `decompose(...)` method."""

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str | None = None,
        max_breadth: int = 6,
    ):
        self.client = client
        self.model = model
        self.max_breadth = max_breadth

    def decompose(
        self,
        problem: str,
        input_summary: str,
        *,
        meter: UsageMeter,
        force_atomic: bool = False,
    ) -> Decomposition:
        """Return the split decision for `problem`. `force_atomic` short-circuits
        to ATOMIC (used at the depth cap) without spending a model call."""
        if force_atomic:
            return Decomposition(atomic=True, reasoning="depth limit reached")

        prompt = _INSTRUCTION.format(problem=problem, input_summary=input_summary)
        msgs = [Message("system", _SYSTEM), Message("user", prompt)]
        resp = self.client.complete(msgs, model=self.model)
        meter.record(resp.usage, sent_messages=msgs, response_text=resp.content)
        data = _extract_json(resp.content)

        if data is None:
            # One repair attempt: re-ask for strict JSON only.
            repair = msgs + [
                Message("assistant", content=resp.content or ""),
                Message("user", "That was not valid JSON. Reply with ONLY the JSON object."),
            ]
            resp = self.client.complete(repair, model=self.model)
            meter.record(resp.usage, sent_messages=repair, response_text=resp.content)
            data = _extract_json(resp.content)

        if not isinstance(data, dict) or data.get("atomic", True):
            return Decomposition(
                atomic=True,
                reasoning=(data or {}).get("reasoning", "") if isinstance(data, dict) else "",
            )

        subs = self._parse_subproblems(data.get("subproblems", []))
        if not subs:
            return Decomposition(atomic=True, reasoning=data.get("reasoning", ""))
        return Decomposition(
            atomic=False, subproblems=subs, reasoning=data.get("reasoning", "")
        )

    def _parse_subproblems(self, raw) -> list[SubProblem]:
        """Validate and normalize the model's sub-problem list."""
        if not isinstance(raw, list):
            return []
        subs: list[SubProblem] = []
        for i, sp in enumerate(raw[: self.max_breadth]):
            if not isinstance(sp, dict):
                continue
            task = str(sp.get("task", "")).strip()
            if not task:
                continue
            sid = str(sp.get("id") or i)
            deps = sp.get("depends_on", [])
            deps = [str(d) for d in deps] if isinstance(deps, list) else []
            subs.append(
                SubProblem(
                    id=sid, task=task, role=str(sp.get("role", "")).strip(), depends_on=deps
                )
            )
        # Drop dependency references to unknown ids (and self-loops) so the
        # executor's topological pass can't deadlock on a bad plan.
        ids = {s.id for s in subs}
        for s in subs:
            s.depends_on = [d for d in s.depends_on if d in ids and d != s.id]
        return subs
