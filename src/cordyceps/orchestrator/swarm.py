"""Swarm — the recursive problem-decomposition orchestrator (Layer 2).

A `Swarm` solves a problem by *decomposing* rather than by doing the work itself:

  1. A `Decomposer` looks at the problem and decides ATOMIC or SPLIT.
  2. ATOMIC  -> a worker `Agent` (the Layer-1 engine) solves it directly and
     returns its answer.
  3. SPLIT   -> the sub-problems form a DAG. Independent ones run in PARALLEL;
     a sub-problem with dependencies starts from its upstream agents' OUTPUT
     (the rest start from the original input). Each sub-problem is itself fed
     back through step 1 — so decomposition is lazy and recursive, and every
     agent is spawned in real time, when its node is reached, with its actual
     resolved input in hand.
  4. The node's answer is the sink of its DAG: a single sink is returned as-is
     (a pipeline's end, or one merge agent's output); multiple sinks are merged
     by a synthesis call.

Safety spine: a depth cap forces ATOMIC at the bottom, and one shared, thread-safe
`UsageMeter` bounds total tokens/calls across the whole tree (root, decomposer
calls, every worker, and synthesis).

    from cordyceps import Swarm
    from cordyceps.capabilities import ShellCapability, FileSystemCapability
    from cordyceps.observability import RunLogger

    swarm = Swarm(capabilities=[ShellCapability(), FileSystemCapability(root="./work")])
    log = RunLogger(live=True)
    answer = swarm.solve("Build and test a small CLI todo app.", on_event=log)
    print(log.render_tree())
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from ..capability import Capability
from ..channel import CallbackChannel, Channel
from ..config import Settings
from ..engine import AgentEngine, Step
from ..llm.base import Message
from ..usage import UsageMeter
from .decomposer import Decomposer, Decomposition, SubProblem


@dataclass
class SwarmResult:
    """Structured outcome of a Swarm run."""

    answer: str
    tokens: int
    calls: int


class _LockedChannel:
    """Serializes a Channel so the parallel DAG nodes can share one sink (the
    RunLogger / CLI printer) without interleaving or racing."""

    def __init__(self, inner: Channel):
        self._inner = inner
        self._lock = threading.Lock()

    def emit(self, step: Step) -> None:
        with self._lock:
            self._inner.emit(step)

    def ask_user(self, question: str) -> str:
        # Held under the lock: a human prompt must not interleave with others.
        with self._lock:
            return self._inner.ask_user(question)


_SYNTH_SYSTEM = (
    "You combine the results of several sub-problems into one coherent, complete "
    "answer to an original problem. Preserve every concrete result; resolve "
    "overlaps; do not invent anything not supported by the sub-results."
)

_WORKER_ROLE_TEMPLATE = (
    "You are a '{role}'. You have been handed ONE scoped sub-problem. Solve it "
    "directly and completely with the tools available, then return a clear, "
    "self-contained result that a coordinating agent can use as-is."
)

_INPUT_HEADER = "\n\n--- Input to work from ---\n"
_DEP_HEADER = "--- Result from sub-problem {sid} ---\n"


class Swarm:
    """The recursive decomposition orchestrator built on the Layer-1 engine."""

    def __init__(
        self,
        *,
        capabilities: list[Capability] | None = None,
        settings: Settings | None = None,
        engine: AgentEngine | None = None,
        decomposer: Decomposer | None = None,
        max_depth: int = 3,
        max_breadth: int = 6,
        max_parallel: int = 4,
        max_total_tokens: int = 0,
        max_total_calls: int = 0,
        worker_planning: bool = False,
    ):
        from ..agent import build_engine  # local import: avoids a cycle

        self.settings = settings or Settings.from_env()
        self.engine = engine or build_engine(self.settings)
        self.capabilities = list(capabilities or [])
        self.decomposer = decomposer or Decomposer(
            self.engine.client, model=self.engine.model, max_breadth=max_breadth
        )
        self.max_depth = max_depth
        self.max_parallel = max(1, max_parallel)
        self.max_total_tokens = max_total_tokens
        self.max_total_calls = max_total_calls
        self.worker_planning = worker_planning

    # -- public entrypoint ----------------------------------------------
    def solve(
        self,
        problem: str,
        *,
        input_text: str | None = None,
        on_event: Callable[[Step], None] | None = None,
        ask_handler: Callable[[str], str] | None = None,
        return_result: bool = False,
    ) -> str | SwarmResult:
        """Decompose and solve `problem`. `input_text` seeds the root agent's
        input (handles for bulk data still travel via a shared DataStore
        capability). Returns the final answer (or a SwarmResult)."""
        meter = UsageMeter(
            max_total_tokens=self.max_total_tokens,
            max_total_calls=self.max_total_calls,
        )
        channel = _LockedChannel(CallbackChannel(on_event=on_event, ask_handler=ask_handler))
        answer = self._solve_node(
            problem, input_text, depth=0, node_id="0", role="solver",
            meter=meter, channel=channel,
        )
        if return_result:
            return SwarmResult(answer=answer, tokens=meter.total_tokens, calls=meter.calls)
        return answer

    # -- the recursive node ---------------------------------------------
    def _solve_node(
        self,
        problem: str,
        input_text: str | None,
        *,
        depth: int,
        node_id: str,
        role: str,
        meter: UsageMeter,
        channel: _LockedChannel,
    ) -> str:
        # At the depth cap (or once the budget is spent) stop splitting and just
        # solve directly — keeps the tree finite and the cost bounded.
        force_atomic = depth >= self.max_depth or meter.exhausted
        decomp = self.decomposer.decompose(
            problem, _summarize_input(input_text), meter=meter, force_atomic=force_atomic
        )
        channel.emit(Step(
            depth=depth, index=0, node_id=node_id,
            note=(
                "atomic" if decomp.atomic
                else f"split into {len(decomp.subproblems)} sub-problems"
            ),
        ))

        if decomp.atomic:
            return self._run_worker(
                problem, input_text, depth=depth, node_id=node_id, role=role,
                meter=meter, channel=channel,
            )
        return self._run_dag(
            problem, decomp, input_text, depth=depth, node_id=node_id,
            meter=meter, channel=channel,
        )

    # -- atomic leaf: a worker agent ------------------------------------
    def _run_worker(
        self, problem, input_text, *, depth, node_id, role, meter, channel
    ) -> str:
        task = problem if not input_text else problem + _INPUT_HEADER + input_text
        instructions = _WORKER_ROLE_TEMPLATE.format(role=role or "solver")
        result = self.engine.run(
            task,
            self.capabilities,
            instructions=instructions,
            depth=depth,
            node_id=node_id,
            channel=channel,
            meter=meter,
            enable_planning=self.worker_planning,
            allow_spawn=False,  # decomposition is the orchestrator's job, not the worker's
        )
        return result if isinstance(result, str) else result.answer

    # -- split node: run the sub-DAG ------------------------------------
    def _run_dag(
        self, problem, decomp: Decomposition, input_text, *, depth, node_id, meter, channel
    ) -> str:
        subs = decomp.subproblems
        child_id = {s.id: f"{node_id}.{i}" for i, s in enumerate(subs)}
        # Announce every child up front so the split tree renders as it forms.
        for s in subs:
            channel.emit(Step(
                depth=depth, index=0, node_id=node_id, spawn=s.task,
                child_id=child_id[s.id],
            ))

        results: dict[str, str] = {}
        done: set[str] = set()
        remaining: dict[str, SubProblem] = {s.id: s for s in subs}

        # Topological waves: run every sub-problem whose dependencies are all
        # satisfied concurrently; repeat until none remain.
        while remaining:
            ready = [s for s in remaining.values() if all(d in done for d in s.depends_on)]
            if not ready:  # malformed/cyclic plan — run the rest rather than hang
                ready = list(remaining.values())

            def run_one(s: SubProblem) -> tuple[str, str]:
                dep_results = [(d, results[d]) for d in s.depends_on if d in results]
                node_input = self._compose_input(input_text, dep_results)
                try:
                    ans = self._solve_node(
                        s.task, node_input, depth=depth + 1, node_id=child_id[s.id],
                        role=s.role or "solver", meter=meter, channel=channel,
                    )
                except Exception as e:  # one node failing must not sink the wave
                    ans = f"[sub-problem {s.id} failed: {type(e).__name__}: {e}]"
                return s.id, ans

            workers = min(len(ready), self.max_parallel)
            if workers == 1:
                wave = [run_one(ready[0])]
            else:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    wave = list(ex.map(run_one, ready))
            for sid, ans in wave:
                results[sid] = ans
                done.add(sid)
                remaining.pop(sid, None)

        answer = self._collect(problem, subs, results, meter=meter)
        channel.emit(Step(depth=depth, index=0, node_id=node_id, final=answer))
        return answer

    # -- combine the DAG's outputs --------------------------------------
    def _collect(self, problem, subs: list[SubProblem], results: dict[str, str], *, meter) -> str:
        """The node's answer = its DAG's sinks. One sink (a pipeline end, or a
        single merge agent) is returned as-is; multiple sinks are synthesized."""
        depended_on = {d for s in subs for d in s.depends_on}
        sinks = [s.id for s in subs if s.id not in depended_on]
        if not sinks:  # cycle fallback: treat everything as a sink
            sinks = [s.id for s in subs]
        if len(sinks) == 1:
            return results.get(sinks[0], "")
        return self._synthesize(problem, [(sid, results.get(sid, "")) for sid in sinks], meter=meter)

    def _synthesize(self, problem, sink_results: list[tuple[str, str]], *, meter) -> str:
        combined = "\n\n".join(
            _DEP_HEADER.format(sid=sid) + ans for sid, ans in sink_results
        )
        msgs = [
            Message("system", _SYNTH_SYSTEM),
            Message(
                "user",
                f"Original problem:\n{problem}\n\nResults from the sub-problems:\n"
                f"{combined}\n\nCombine these into one complete, coherent answer "
                "to the original problem.",
            ),
        ]
        resp = self.engine.client.complete(msgs, model=self.engine.model)
        meter.record(resp.usage, sent_messages=msgs, response_text=resp.content)
        return (resp.content or "").strip() or combined

    @staticmethod
    def _compose_input(base_input: str | None, dep_results: list[tuple[str, str]]) -> str | None:
        """A node with dependencies starts from its upstream agents' output; a
        node with none starts from the original input."""
        if dep_results:
            return "\n\n".join(_DEP_HEADER.format(sid=sid) + ans for sid, ans in dep_results)
        return base_input


def _summarize_input(input_text: str | None) -> str:
    """A short, context-cheap description of a node's input for the decomposer —
    the raw bulk never enters the planning call."""
    if not input_text:
        return "the original problem statement (no prior results yet)"
    text = input_text.strip()
    if len(text) <= 280:
        return text
    return f"{len(text)} chars of prior results, beginning: {text[:240]}…"
