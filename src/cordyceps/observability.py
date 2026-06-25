"""Run observability — capture and visualize how a run unfolds.

`RunLogger` is an `on_event` callback you hand to `Agent.solve(...)`. It does two
things:

  1. **Logs everything.** Every `Step` the engine emits — across all depths and
     of every kind (code, output, llm_call, spawn, usage, compaction,
     interaction, final, note) — is recorded, and optionally streamed to a JSONL
     file. Nothing about the run is dropped.

  2. **Shows the split.** It reconstructs the decomposition tree from the
     node ids the engine stamps on each event (`0`, `0.0`, `0.1`, `0.1.0`, …) and
     can print it live as agents spawn, and as a final tree at the end.

    from cordyceps import Agent
    from cordyceps.observability import RunLogger

    log = RunLogger(jsonl_path="run.jsonl", live=True)
    agent.solve(task, on_event=log)
    print(log.render_tree())   # the full decomposition, after the run
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any, TextIO

from .engine import Step


def _short(text: str | None, limit: int) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class Node:
    """One agent invocation in the decomposition tree."""

    node_id: str
    task: str = ""              # the subtask it was spawned with (root: the goal)
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    result: str | None = None  # its final answer
    steps: int = 0
    code_blocks: int = 0
    llm_calls: int = 0
    tokens: int = 0            # cumulative tokens at this node's last usage event


class RunLogger:
    """An on_event sink: records every Step, writes JSONL, renders the split tree."""

    def __init__(
        self,
        *,
        jsonl_path: str | None = None,
        live: bool = True,
        verbose: bool = False,
        stream: TextIO | None = None,
        value_limit: int = 120,
    ):
        self.events: list[dict[str, Any]] = []
        self.nodes: dict[str, Node] = {}
        self.root_id: str | None = None
        self.value_limit = value_limit
        self.live = live
        # verbose also streams each code block and its output (not just the
        # spawn/final split). Useful for watching exactly what an agent runs.
        self.verbose = verbose
        self._out = stream or sys.stderr
        self._fh = open(jsonl_path, "w") if jsonl_path else None

    # -- the on_event callback ------------------------------------------
    def __call__(self, step: Step) -> None:
        self._record(step)
        node = self._ensure_node(step.node_id or "0")
        if self.root_id is None:
            self.root_id = node.node_id
            if not node.task:
                node.task = "(root task)"
        node.steps += 1

        if step.spawn and step.child_id:
            child = self._ensure_node(step.child_id)
            child.parent_id = node.node_id
            child.task = step.spawn
            if step.child_id not in node.children:
                node.children.append(step.child_id)
            if self.live:
                self._print_split(node.node_id, child)
        if step.code:
            node.code_blocks += 1
            if self.live and self.verbose:
                self._print_block(node.node_id, "»", step.code)
        if step.output is not None and self.live and self.verbose:
            self._print_block(node.node_id, "«", step.output)
        if step.llm_call:
            node.llm_calls += 1
        if step.usage:
            node.tokens = step.usage.cumulative_tokens
        if step.final is not None:
            node.result = step.final
            if self.live:
                self._print_final(node)

    # -- live printing --------------------------------------------------
    def _indent(self, node_id: str) -> str:
        return "  " * node_id.count(".")

    def _print_split(self, parent_id: str, child: Node) -> None:
        ind = self._indent(child.node_id)
        print(f"{ind}↳ split [{child.node_id}] {_short(child.task, self.value_limit)}",
              file=self._out, flush=True)

    def _print_final(self, node: Node) -> None:
        ind = self._indent(node.node_id)
        print(f"{ind}✓ [{node.node_id}] → {_short(node.result, self.value_limit)}",
              file=self._out, flush=True)

    def _print_block(self, node_id: str, glyph: str, text: str) -> None:
        ind = self._indent(node_id)
        body = (text or "").rstrip()
        first = True
        for line in body.splitlines() or [""]:
            mark = f"{ind}{glyph} " if first else f"{ind}  "
            print(mark + line, file=self._out, flush=True)
            first = False

    # -- recording ------------------------------------------------------
    def _record(self, step: Step) -> None:
        kinds = [
            k for k in ("code", "output", "final", "message", "spawn", "note")
            if getattr(step, k) is not None
        ]
        if step.llm_call:
            kinds.append("llm_call")
        if step.usage:
            kinds.append("usage")
        if step.compaction:
            kinds.append("compaction")
        if step.interaction:
            kinds.append("interaction")
        if step.todo is not None:
            kinds.append("todo")
        row = {
            "node_id": step.node_id,
            "depth": step.depth,
            "index": step.index,
            "kinds": kinds,
            "child_id": step.child_id,
            "spawn": step.spawn,
            "code": step.code,
            "output": step.output,
            "final": step.final,
            "message": step.message,
            "note": step.note,
            "cumulative_tokens": step.usage.cumulative_tokens if step.usage else None,
        }
        self.events.append(row)
        if self._fh is not None:
            self._fh.write(json.dumps(row) + "\n")
            self._fh.flush()

    def _ensure_node(self, node_id: str) -> Node:
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(node_id=node_id)
        return self.nodes[node_id]

    # -- after-the-run views --------------------------------------------
    def render_tree(self, value_limit: int | None = None) -> str:
        """A static rendering of the whole decomposition tree."""
        limit = value_limit or self.value_limit
        lines: list[str] = []

        def walk(node_id: str) -> None:
            node = self.nodes[node_id]
            ind = "  " * node_id.count(".")
            head = f"{ind}[{node_id}] {_short(node.task, limit)}"
            meta = f"  ({node.steps} steps, {node.code_blocks} code, {node.tokens} tok)"
            lines.append(head + meta)
            if node.result is not None:
                lines.append(f"{ind}  → {_short(node.result, limit)}")
            for child_id in node.children:
                walk(child_id)

        if self.root_id is not None:
            walk(self.root_id)
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        """Counts useful for asserting on a run in tests / dashboards."""
        return {
            "nodes": len(self.nodes),
            "events": len(self.events),
            "max_depth": max((nid.count(".") for nid in self.nodes), default=0),
            "total_tokens": max((n.tokens for n in self.nodes.values()), default=0),
        }

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
