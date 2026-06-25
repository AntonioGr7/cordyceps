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

## The orchestrator: `Swarm`

Where an `Agent` *does* the work, a `Swarm` *decomposes* it. You hand it a problem;
a **decomposer** agent splits it into sub-problems with dependency edges, the sub-DAG
runs (independent sub-problems in parallel, dependent ones piping output→input), and
each sub-problem is itself re-decomposed — lazily, only when it's reached — until it's
simple enough for a worker `Agent` to solve directly. It's a **tree-of-DAGs**: the
recursion is the tree, each level is a sub-DAG.

```python
from cordyceps import Swarm, RunLogger
from cordyceps.capabilities import ShellCapability, FileSystemCapability

swarm = Swarm(
    capabilities=[ShellCapability(cwd="./work"), FileSystemCapability(root="./work")],
    max_depth=3,      # how deep decomposition may recurse
    max_breadth=6,    # max sub-problems per split
    max_parallel=4,   # independent sub-problems run concurrently
    max_total_tokens=400_000,
)
log = RunLogger(live=True)
result = swarm.solve("Build a wc-lite CLI, write tests for it, run them, and report.",
                     on_event=log, return_result=True)
print(log.render_tree())   # the decomposition tree, after the run
print(result.answer)
```

### How a `Swarm` run unfolds

`solve()` creates one shared, thread-safe `UsageMeter` (the global token/call budget
for the whole tree) and a serialized event channel, then calls `_solve_node` on the
root. Every node runs the same recursive procedure:

1. **Decide.** The `Decomposer` makes one structured LLM call returning `ATOMIC` or a
   list of sub-problems with `depends_on` edges. It sees only a *short summary* of the
   node's input, never the raw bulk. At the depth cap (or once the budget is spent) the
   decision is forced to `ATOMIC` with **no** call — this is what guarantees termination.
2. **Atomic → solve.** A worker `Agent` runs on the Layer-1 engine with the node's
   input as its task and a role-specific identity. Workers run with `allow_spawn=False`:
   decomposition is the orchestrator's job, so a worker never re-splits — it just solves.
3. **Split → run the sub-DAG.** Sub-problems are scheduled in **topological waves**:
   every sub-problem whose dependencies are all done runs in the same wave, concurrently
   (`max_parallel`). A node with dependencies starts from its **upstream agents' output**;
   a node with none starts from the **original input**. Each sub-problem recurses back
   into step 1 (this is the "split again only if still complex" behavior).
4. **Collect.** The node's answer is its DAG's *sinks* (nodes nothing depends on). A
   single sink — a pipeline's end — is returned as-is; multiple sinks are merged by one
   synthesis call.

Safety spine: a **depth cap** forces `ATOMIC` at the bottom, and the **shared budget**
bounds total cost across the root, every decomposer call, every worker, and synthesis.
`RunLogger` stamps each event with a node id (`0`, `0.0`, `0.1.2`, …) and reconstructs
the whole split tree live.

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

**Layer 2 — the Cordyceps orchestrator (this repo, now).** A thin layer on top: a
`Decomposer` agent that, given a problem, returns either *atomic* or a **sub-DAG**
of sub-problems with dependency edges. An executor (`Swarm`) runs the DAG
topologically (independent nodes in parallel, dependent nodes piping output→input);
each node recurses back into the decomposer — a **tree-of-DAGs** (lazy recursion).
Leaves are Layer-1 `Agent`s. See *The orchestrator: `Swarm`* above for the run flow.

- `cordyceps/orchestrator/decomposer.py` — the `Decomposer` (structured split call)
- `cordyceps/orchestrator/swarm.py` — the `Swarm` executor (waves, edges, synthesis)

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
python examples/basic_agent.py      # single Agent; needs ANTHROPIC_API_KEY
python examples/swarm_decompose.py  # the Swarm orchestrator; needs ANTHROPIC_API_KEY
```

## Status / notes

- Execution is **in-process** (`exec`) — fast and full-power, **not** sandboxed.
  A container/gVisor backend slots in behind the `ExecutionEnvironment` ABC.
- Extended thinking is off by default in the Anthropic client (replaying thinking
  blocks across turns needs verbatim echo-back; reasoning depth is steered with
  `effort` instead). Thinking-block round-tripping is a planned extension.
