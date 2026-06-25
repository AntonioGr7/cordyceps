# Cordyceps

A recursive, code-action agent and problem-decomposition substrate.

The name is from *The Last of Us* — the fungus that spreads through a host. Here
the "spread" is controlled: an agent splits a problem into sub-problems and spawns
a sub-agent per piece, recursively, bounded by depth and budget.

## The base unit: `Agent`

The foundation is a **SOTA standalone agent** whose only action is running Python
in a persistent REPL. Everything it can touch — the shell/OS, the filesystem,
arbitrary tools/MCP, data sources — is exposed as ordinary **callables in that
REPL**. Because the action is code, the agent composes tools with loops and
conditionals in a single step, instead of one round-trip per call from a flat JSON
tool menu (the CodeAct insight). It decomposes via `spawn(subtask)`, which
re-enters a fresh sub-agent and returns just its answer — keeping sub-work out of
the parent's context.

```python
from cordyceps import Agent
from cordyceps.capabilities import ShellCapability, FileSystemCapability

agent = Agent(
    role="builder",
    description="You implement small, well-scoped programming tasks.",
    capabilities=[ShellCapability(), FileSystemCapability(root="./work")],
)
print(agent.solve("Create hello.py that prints 'hi' and run it."))
```

## Architecture

**Layer 1 — `Agent` (this repo, now).** The code-action engine generalized from a
RAG loop ([vomero](https://github.com/AntonioGr7/vomero)) into a general-purpose
agent: one `python` tool, a set of `Capability` objects mounted into the REPL,
core helpers (`llm`, `llm_batched`, `spawn`, `answer`), context compaction, a
shared token/call budget, and a depth cap.

- `cordyceps/engine/loop.py` — the recursive loop (`AgentEngine`)
- `cordyceps/capability.py` — the `Capability` seam (`bind` + `surface`)
- `cordyceps/capabilities/` — shell, filesystem, tool-registry
- `cordyceps/llm/` — provider-agnostic client; native Anthropic (`claude-opus-4-8`)
- `cordyceps/agent.py` — the `Agent` role contract

**Layer 2 — the Cordyceps orchestrator (next).** A thin layer on top: a
`Decomposer` agent that, given a problem, returns either *atomic* or a **sub-DAG**
of sub-problems with dependency edges. An executor runs the DAG topologically
(independent nodes in parallel, dependent nodes piping output→input); each node is
a Layer-1 `Agent` that may itself decompose — a **tree-of-DAGs** (lazy recursion).

## Configuration

Reads the environment (and a local `.env`). Key vars:

- `ANTHROPIC_API_KEY` — required for the default provider
- `CORDYCEPS_MODEL` (default `claude-opus-4-8`), `CORDYCEPS_EFFORT`
- `CORDYCEPS_MAX_DEPTH`, `CORDYCEPS_MAX_TOTAL_TOKENS`, `CORDYCEPS_MAX_TOTAL_CALLS`
- `CORDYCEPS_PROVIDER` — `anthropic` (default) or `openai` (any compatible endpoint)

## Develop

```bash
uv venv && uv pip install -e ".[dev]"
pytest            # runs against a scripted fake client — no API key needed
python examples/basic_agent.py   # needs ANTHROPIC_API_KEY
```

## Status / notes

- Execution is **in-process** (`exec`) — fast and full-power, **not** sandboxed.
  A container/gVisor backend slots in behind the `ExecutionEnvironment` ABC.
- Extended thinking is off by default in the Anthropic client (replaying thinking
  blocks across turns needs verbatim echo-back; reasoning depth is steered with
  `effort` instead). Thinking-block round-tripping is a planned extension.
