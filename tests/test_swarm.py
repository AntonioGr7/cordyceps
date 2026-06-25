"""Swarm (Layer 2) tests with a content-routing fake client — no network.

The DAG runs nodes in parallel, so call ORDER is nondeterministic; the fake
client therefore responds based on message CONTENT (which kind of call, which
sub-problem) rather than a fixed queue. Markers in the problem/task strings drive
the routing.
"""

from __future__ import annotations

import json
import threading

from cordyceps.config import Settings
from cordyceps.engine import AgentEngine
from cordyceps.engine.loop import build_system_prompt
from cordyceps.llm.base import LLMResponse, ToolCall, Usage
from cordyceps.observability import RunLogger
from cordyceps.orchestrator import Swarm


class RoutingClient:
    """Deterministic per-call responses keyed on message content (thread-safe)."""

    def __init__(self):
        self.calls = 0
        self._lock = threading.Lock()

    def complete(self, messages, *, tools=None, model=None, temperature=None):
        with self._lock:
            self.calls += 1
        system = " ".join(m.content or "" for m in messages if m.role == "system")
        user = " ".join(m.content or "" for m in messages if m.role == "user")

        if tools:  # a worker agent's code-action loop
            return LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="t", name="python",
                                     arguments={"code": self._worker_code(user)})],
                usage=Usage(5, 5),
            )
        if "decomposition planner" in system:
            return LLMResponse(content=self._plan(user), usage=Usage(5, 5))
        if "combine the results of several sub-problems" in system:
            return LLMResponse(content="MERGED", usage=Usage(5, 5))
        return LLMResponse(content="ok", usage=Usage(1, 1))

    @staticmethod
    def _plan(user: str) -> str:
        if "ROOTQ" in user:  # pipeline: b depends on a's output
            return json.dumps({"atomic": False, "subproblems": [
                {"id": "a", "task": "TASK_A", "role": "ra", "depends_on": []},
                {"id": "b", "task": "TASK_B", "role": "rb", "depends_on": ["a"]},
            ]})
        if "FANOUT" in user:  # two independent sinks -> synthesized
            return json.dumps({"atomic": False, "subproblems": [
                {"id": "c", "task": "TASK_C", "depends_on": []},
                {"id": "d", "task": "TASK_D", "depends_on": []},
            ]})
        return json.dumps({"atomic": True})  # leaves

    @staticmethod
    def _worker_code(user: str) -> str:
        if "TASK_A" in user:
            return "answer('RES_A')"
        if "TASK_B" in user:  # verify the dependency's output reached this node
            return "answer('RES_B_GOT_A')" if "RES_A" in user else "answer('RES_B_NO_INPUT')"
        if "TASK_C" in user:
            return "answer('RES_C')"
        if "TASK_D" in user:
            return "answer('RES_D')"
        return "answer('RES_DEFAULT')"


def _swarm(client, **kw) -> Swarm:
    engine = AgentEngine(client, max_steps=4, max_depth=3)
    return Swarm(engine=engine, settings=Settings(), **kw)


def test_pipeline_passes_upstream_output_to_dependent_node():
    client = RoutingClient()
    swarm = _swarm(client, max_depth=3)
    # a -> b: b is the only sink, so the answer is b's output, and b must have
    # received a's result as its input.
    assert swarm.solve("ROOTQ: do the thing") == "RES_B_GOT_A"


def test_fanout_synthesizes_multiple_sinks():
    client = RoutingClient()
    swarm = _swarm(client, max_depth=3)
    assert swarm.solve("FANOUT: do two things") == "MERGED"


def test_depth_cap_forces_atomic_worker_without_planning():
    client = RoutingClient()
    swarm = _swarm(client, max_depth=0)
    # depth 0 >= max_depth 0 -> no decomposition call; solved directly.
    result = swarm.solve("ROOTQ: do the thing", return_result=True)
    assert result.answer == "RES_DEFAULT"


def test_split_tree_is_observable():
    client = RoutingClient()
    swarm = _swarm(client, max_depth=3)
    log = RunLogger(live=False)
    answer = swarm.solve("ROOTQ: do the thing", on_event=log)
    assert answer == "RES_B_GOT_A"
    assert log.root_id == "0"
    assert log.nodes["0"].children == ["0.0", "0.1"]
    assert log.nodes["0.0"].task == "TASK_A"
    assert log.nodes["0.1"].task == "TASK_B"
    assert log.nodes["0.0"].result == "RES_A"


def test_worker_prompt_withholds_spawn():
    with_spawn = build_system_prompt([], allow_spawn=True)
    without = build_system_prompt([], allow_spawn=False)
    assert "spawn(" in with_spawn
    assert "spawn(" not in without
