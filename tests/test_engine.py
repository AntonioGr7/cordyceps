"""Engine tests with a scripted fake client — no network, no API key.

These exercise the code-action loop, capability injection, and the spawn
primitive without touching a real provider.
"""

from __future__ import annotations

from cordyceps.capabilities import ToolRegistry, Tool
from cordyceps.engine import AgentEngine
from cordyceps.engine.loop import build_system_prompt
from cordyceps.llm.base import LLMResponse, ToolCall, Usage


class ScriptedClient:
    """Returns a queued list of LLMResponses, one per complete() call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def complete(self, messages, *, tools=None, model=None, temperature=None):
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        # Default: end the turn with a plain reply.
        return LLMResponse(content="(no more scripted responses)", usage=Usage(1, 1))


def _code_step(code: str) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="t1", name="python", arguments={"code": code})],
        usage=Usage(10, 5),
    )


def test_basic_answer_loop():
    client = ScriptedClient([_code_step("answer('done')")])
    engine = AgentEngine(client, max_steps=3)
    result = engine.run("do the thing", [])
    assert result == "done"


def test_capability_tool_is_callable_in_repl():
    registry = ToolRegistry([Tool(name="double", func=lambda x: x * 2)])
    client = ScriptedClient([_code_step("answer(double(21))")])
    engine = AgentEngine(client, max_steps=3)
    result = engine.run("double 21", [registry])
    assert result == "42"


def test_repl_state_persists_across_steps():
    client = ScriptedClient([
        _code_step("x = 40"),
        _code_step("answer(x + 2)"),
    ])
    engine = AgentEngine(client, max_steps=5)
    assert engine.run("compute", []) == "42"


def test_plain_text_reply_is_final():
    client = ScriptedClient([LLMResponse(content="just an answer", usage=Usage(1, 1))])
    engine = AgentEngine(client, max_steps=3)
    assert engine.run("hi", []) == "just an answer"


def test_spawn_recurses_and_returns_subanswer():
    # Root spawns a sub-task; the sub-agent answers; root answers with that.
    client = ScriptedClient([
        _code_step("sub = spawn('sub problem')"),       # root step 0 (depth 0)
        _code_step("answer('child says nothing useful')"),  # child step 0 (depth 1)
        _code_step("answer(sub)"),                        # root step 1 (depth 0)
    ])
    engine = AgentEngine(client, max_steps=5, max_depth=2)
    result = engine.run("parent task", [])
    assert result == "child says nothing useful"


def test_traceback_is_fed_back_then_recovered():
    client = ScriptedClient([
        _code_step("answer(undefined_name)"),  # raises NameError
        _code_step("answer('recovered')"),
    ])
    engine = AgentEngine(client, max_steps=5)
    assert engine.run("x", []) == "recovered"


def test_budget_exhaustion_stops_cleanly():
    # max_total_calls=1 means after the first model call the meter is exhausted.
    from cordyceps.usage import UsageMeter

    client = ScriptedClient([_code_step("x = 1")] * 10)
    engine = AgentEngine(client, max_steps=10)
    meter = UsageMeter(max_total_calls=1)
    result = engine.run("x", [], meter=meter)
    assert isinstance(result, str)
    assert client.calls == 1


def test_system_prompt_includes_capability_surface():
    registry = ToolRegistry([Tool(name="search", func=lambda q: q, description="Search the web.")])
    prompt = build_system_prompt([registry], instructions="You are a tester.")
    assert "search" in prompt
    assert "Search the web." in prompt
    assert "You are a tester." in prompt
    assert "spawn(" in prompt  # core helpers present


def test_datastore_shared_across_spawn_tree():
    """Seed input on the host, a sub-agent computes and stores a result, the
    parent reads it, and the host collects the output — all without data ever
    entering a task string or a spawn return value."""
    from cordyceps.capabilities import DataStore

    store = DataStore({"input": [1, 2, 3, 4]})
    client = ScriptedClient([
        _code_step("k = spawn('sum the input')"),                       # root depth 0
        _code_step("put_data('partial', sum(data('input'))); answer('stored under partial')"),  # child depth 1
        _code_step("answer(data('partial'))"),                          # root depth 0
    ])
    engine = AgentEngine(client, max_steps=5, max_depth=2)
    result = engine.run("compute the sum of the input", [store])
    assert result == "10"
    assert store.get("partial") == 10  # host collects the output


def test_node_ids_and_logger_reconstruct_split_tree(tmp_path):
    """Every event is stamped with a node id; RunLogger rebuilds the split tree
    and writes a complete JSONL log."""
    from cordyceps.observability import RunLogger

    client = ScriptedClient([
        _code_step("a = spawn('analyze A')"),   # root 0 spawns child 0.0
        _code_step("answer('A done')"),          # child 0.0
        _code_step("b = spawn('analyze B')"),    # root 0 spawns child 0.1
        _code_step("answer('B done')"),          # child 0.1
        _code_step("answer('combined')"),        # root 0
    ])
    engine = AgentEngine(client, max_steps=8, max_depth=2)
    log = RunLogger(jsonl_path=str(tmp_path / "run.jsonl"), live=False)
    result = engine.run("compare A and B", [], on_event=log)
    log.close()

    assert result == "combined"
    # tree shape: root 0 with two children 0.0 and 0.1
    assert log.root_id == "0"
    assert log.nodes["0"].children == ["0.0", "0.1"]
    assert log.nodes["0.0"].task == "analyze A"
    assert log.nodes["0.0"].result == "A done"
    assert log.nodes["0.1"].result == "B done"
    assert log.summary()["nodes"] == 3
    assert log.summary()["max_depth"] == 1
    # every event carries a node id, and the JSONL has a line per event
    assert all(e["node_id"] is not None for e in log.events)
    lines = (tmp_path / "run.jsonl").read_text().strip().splitlines()
    assert len(lines) == len(log.events)
