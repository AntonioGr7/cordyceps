"""AgentEngine — the recursive code-action loop at the heart of Cordyceps.

One `run` mounts a set of `Capability` objects into a persistent Python REPL,
hands the model a single tool (`python`), and loops:

    model -> python(code) -> exec -> stdout/traceback fed back -> repeat

until the model calls `answer(...)` from the REPL (or replies in plain text).

The model's only lever on the world is code. Capabilities (shell, filesystem,
tools/MCP, data sources) appear as ordinary callables in that REPL, so the model
composes them with normal control flow instead of a one-call-per-step JSON menu.

Recursion is the decomposition primitive: `spawn(subtask, scope=None)` re-enters
this engine at depth+1 with the same capabilities (optionally scoped), so a
self-contained sub-problem gets the full power of a fresh agent and returns just
its answer — keeping the sub-work out of the parent's context. Depth and a shared
token/call budget keep the recursion finite.

The engine holds no per-run state: usage flows through a caller-owned UsageMeter,
so one engine instance safely serves concurrent runs (and its own recursion).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

from ..capability import Capability, CapabilityContext
from ..channel import Channel, CallbackChannel
from ..execution import ExecutionEnvironment, InProcessEnvironment
from ..llm.base import LLMClient, Message, ToolSpec
from ..usage import UsageMeter, UsageSnapshot, estimate_message_tokens
from .compaction import Compactor, CompactionEvent
from .todo import TodoItem, TodoList

# Returned from llm()/spawn() when the global budget is spent, so the model's
# code keeps running (and can call answer()) instead of erroring mid-step.
_BUDGET_NOTICE = (
    "[budget exhausted: this sub-call was skipped to keep the run within its "
    "token/call limit. Answer now with what you already have.]"
)

_PARTIAL_SYNTHESIS_PROMPT = (
    "You have reached the step limit and cannot do any more work or tool "
    "calls. Using ONLY what you have already gathered above, write a brief, "
    "honest reply that:\n"
    "1. States up front that the step limit was reached, so this answer is "
    "partial and may be incomplete.\n"
    "2. Organizes and presents the relevant findings you did gather so far.\n"
    "3. Notes briefly what is still unresolved.\n"
    "If nothing useful was found, say that plainly instead of guessing."
)

# The single tool the model gets. One tool (run Python) maps cleanly onto every
# provider's function-calling and embodies the code-action idea.
PYTHON_TOOL = ToolSpec(
    name="python",
    description=(
        "Run Python in a persistent REPL to do your work and build your answer. "
        "State (variables, imports) persists across calls. Use print() to see "
        "anything. Call answer(value) when you are done."
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source to execute."}
        },
        "required": ["code"],
    },
)

SYSTEM_PROMPT_HEAD = """You are an autonomous agent that solves a task by writing \
and running Python in a persistent REPL. Code is your ONLY way to act: you think, \
then you run code, observe its output, and continue until the task is done.

Your REPL already has these names available:

"""

# Core helpers always present, regardless of which capabilities are mounted.
# Split so the spawn helper can be withheld for non-recursing worker agents (the
# orchestrator owns decomposition there) — see build_system_prompt(allow_spawn).
CORE_HELPERS_DISTILL = """  llm(text, system=None) -> str
                A single, fresh model call with NO memory and NO tools. Use it to
                distill a chunk of text you already have (summarize, extract,
                classify). The text you pass is the ONLY thing that sub-call sees.

  llm_batched(texts, system=None) -> list[str]
                Like llm(), but runs MANY distillations concurrently, results in
                input order. The partition+map workhorse for processing many
                chunks in parallel.
"""

CORE_HELPERS_SPAWN = """  spawn(subtask, scope=None) -> str
                Delegate a self-contained SUB-PROBLEM to a fresh sub-agent. It
                gets the same capabilities you have — its own REPL, its own
                reasoning — and returns just its final answer, so its work stays
                OUT of your context. This is how you DECOMPOSE: split the task
                into independent or sequential sub-problems and spawn one per
                piece, passing each the input it needs (often the previous
                sub-agent's result). `scope` optionally narrows any data-source
                capability for the sub-agent. Spawn only when a sub-problem is
                genuinely complex enough to warrant its own agent; do simple
                steps inline.
"""

CORE_HELPERS_ANSWER = """  answer(value) Record your FINAL answer and finish. `value` may be a string OR
                any REPL variable — pass a variable you built up and its full
                contents become the answer (not limited by your own output size).
"""

STRATEGY = """
Strategy:
  - Plan first: decide whether the task is atomic or should be decomposed. If
    it decomposes, name the sub-problems and their dependencies (which feed into
    which), then resolve them — independent ones can be handled in any order,
    dependent ones in sequence, passing results forward.
  - Delegate a sub-problem with spawn(...) ONLY when it is substantial enough to
    deserve its own agent. For small steps, just write the code inline.
  - Keep raw bulk OUT of your context: hold large text/data in variables and pass
    them to llm()/llm_batched()/spawn() to distill, rather than printing it all.
  - Verify before finishing: check that your result actually satisfies the task,
    and that each sub-result you relied on is sound.
  - When confident, call answer(...) with your result.

Keep each code block small and purposeful. Print only what you need to see."""

# Strategy for a worker agent that may NOT spawn: it solves its scoped sub-problem
# directly (decomposition is the orchestrator's job, not the worker's).
STRATEGY_LEAF = """
Strategy:
  - You have been handed one scoped sub-problem to solve directly — do NOT try to
    split it further; solve it with code here.
  - Keep raw bulk OUT of your context: hold large text/data in variables and pass
    them to llm()/llm_batched() to distill, rather than printing it all.
  - Verify before finishing: check that your result actually satisfies the task.
  - When confident, call answer(...) with your result.

Keep each code block small and purposeful. Print only what you need to see."""

PLANNING_PROMPT = """

You also have a TODO surface to externalize your plan — the user watches it live:

  todo.plan([...])   Set your step-by-step plan. Call this FIRST.
  todo.start(n)      Mark item n (1-based) in progress, right before you work on it.
  todo.complete(n)   Mark item n done, right after you finish it.
  todo.add("...")    Append a step you discover mid-task.

Keep the plan to a handful of concrete, verifiable steps. You do NOT need to
print the list; it is shown to the user automatically."""

ASK_USER_PROMPT = """

You can ask the user for help when genuinely stuck:

  ask_user(question) -> str   Pause and ask the user; returns their reply.

Use it SPARINGLY — only when proceeding would mean guessing on something that
matters. Ask one specific question at a time."""

ASK_PARENT_PROMPT = """

You are a sub-task delegated by a parent agent that holds the broader goal. When
you lack context to proceed, ask it first:

  ask_parent(question) -> str   Ask the delegating agent; returns its reply.

Prefer this over asking the user for anything about the task's intent or scope."""


def build_system_prompt(
    capabilities: list[Capability],
    *,
    instructions: str | None = None,
    extra: str | None = None,
    allow_spawn: bool = True,
) -> str:
    """Assemble the root system prompt from the head, each capability's surface,
    the core helpers, and the strategy. `instructions` is the agent's role/task
    contract; `extra` is an optional tunable block. When `allow_spawn` is False
    the spawn helper and the decomposition strategy are withheld — used for
    worker agents the orchestrator drives, which solve one sub-problem directly."""
    parts = [SYSTEM_PROMPT_HEAD, CORE_HELPERS_DISTILL]
    if allow_spawn:
        parts.append(CORE_HELPERS_SPAWN)
    parts.append(CORE_HELPERS_ANSWER)
    for cap in capabilities:
        surface = cap.surface()
        if surface and surface.strip():
            parts.append("\n" + surface.rstrip() + "\n")
    parts.append(STRATEGY if allow_spawn else STRATEGY_LEAF)
    prompt = "".join(parts)
    if instructions and instructions.strip():
        prompt += "\n\nYour role and task:\n" + instructions.strip()
    if extra and extra.strip():
        prompt += "\n\nAdditional instructions:\n" + extra.strip()
    return prompt


def truncate_output(text: str, limit: int) -> str:
    """Cap a tool result before it enters the transcript, keeping head+tail."""
    if limit <= 0 or len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    dropped = len(text) - head - tail
    marker = (
        f"\n\n…[output truncated: {dropped:,} of {len(text):,} chars elided. "
        "Don't print large values wholesale — slice them, or hold them in a "
        "variable and pass to llm()/spawn() to distill.]…\n\n"
    )
    return text[:head] + marker + text[-tail:]


@dataclass
class LLMCall:
    """A flat llm() distillation sub-call, surfaced for the trace."""

    prompt: str
    response: str
    tokens: int


@dataclass
class Interaction:
    """A round-trip where the agent asked for help and got a reply.
    `kind` is "user" or "parent"."""

    question: str
    answer: str
    kind: str = "user"


@dataclass
class Step:
    """One turn in the loop — handed to the channel for observability."""

    depth: int
    index: int
    code: str | None = None
    output: str | None = None
    final: str | None = None
    message: str | None = None
    todo: list[TodoItem] | None = None
    interaction: Interaction | None = None
    usage: UsageSnapshot | None = None
    compaction: CompactionEvent | None = None
    llm_call: LLMCall | None = None
    note: str | None = None
    # Set when a sub-agent is spawned, for rendering the recursion tree.
    spawn: str | None = None
    # Dotted identity of the agent node this event belongs to ("0", "0.1", …).
    node_id: str | None = None
    # On a spawn step: the node_id assigned to the child being spawned. The edge
    # (node_id -> child_id) plus the child's `spawn` text reconstructs the split.
    child_id: str | None = None


@dataclass
class RunResult:
    """Structured outcome of a root run(..., return_trajectory=True)."""

    answer: str
    trajectory: list[Step]
    tokens: int
    calls: int


class _TrajectoryRecorder:
    """Channel wrapper that captures depth-0 Steps while delegating every event
    to the real channel."""

    def __init__(self, inner: Channel):
        self.inner = inner
        self.steps: list[Step] = []

    def emit(self, step: Step) -> None:
        if step.depth == 0:
            self.steps.append(step)
        self.inner.emit(step)

    def ask_user(self, question: str) -> str:
        return self.inner.ask_user(question)


class AgentEngine:
    """The stateless, reentrant code-action loop. One instance serves concurrent
    runs and its own recursion; per-run state lives in the caller-owned meter and
    the locals of `run`."""

    def __init__(
        self,
        client: LLMClient,
        *,
        env_factory: Callable[[], ExecutionEnvironment] = InProcessEnvironment,
        model: str | None = None,
        max_steps: int = 24,
        max_depth: int = 3,
        max_output_chars: int = 10_000,
        max_parallel_calls: int = 8,
        extra_instructions: str | None = None,
        compactor: Compactor | None = None,
        enable_planning: bool = False,
        planning_root_only: bool = False,
        enable_interaction: bool = False,
        interaction_root_only: bool = False,
        workspace: str | None = None,
    ):
        self.client = client
        self.env_factory = env_factory
        self.model = model
        self.max_steps = max_steps
        self.max_depth = max_depth
        self.max_output_chars = max_output_chars
        self.max_parallel_calls = max_parallel_calls
        self.extra_instructions = extra_instructions
        self.compactor = compactor
        self.enable_planning = enable_planning
        self.planning_root_only = planning_root_only
        self.enable_interaction = enable_interaction
        self.interaction_root_only = interaction_root_only
        self.workspace = workspace

    def run(
        self,
        task: str,
        capabilities: list[Capability] | None = None,
        *,
        instructions: str | None = None,
        depth: int = 0,
        node_id: str = "0",
        channel: Channel | None = None,
        meter: UsageMeter | None = None,
        clarify_handler: Callable[[str], str] | None = None,
        enable_planning: bool | None = None,
        planning_root_only: bool | None = None,
        env: ExecutionEnvironment | None = None,
        on_event: Callable[[Step], None] | None = None,
        ask_handler: Callable[[str], str] | None = None,
        return_trajectory: bool = False,
        allow_spawn: bool = True,
    ) -> str | RunResult:
        capabilities = list(capabilities or [])
        if channel is None:
            channel = CallbackChannel(on_event=on_event, ask_handler=ask_handler)
        recorder = _TrajectoryRecorder(channel) if return_trajectory else None
        if recorder is not None:
            channel = recorder
        meter = meter or UsageMeter()

        def emit(step: Step) -> None:
            # Stamp every event with this agent node's identity, so the whole
            # split tree is reconstructable from the event stream alone.
            step.node_id = node_id
            channel.emit(step)

        def _finish(ans: str):
            if recorder is None:
                return ans
            return RunResult(
                answer=ans,
                trajectory=recorder.steps,
                tokens=meter.total_tokens,
                calls=meter.calls,
            )

        if enable_planning is None:
            enable_planning = self.enable_planning
        if planning_root_only is None:
            planning_root_only = self.planning_root_only

        env = env if env is not None else self.env_factory()
        holder: dict[str, str] = {}
        current: dict[str, int] = {"index": 0}
        spawned: dict[str, int] = {"n": 0}  # children spawned by this node so far
        pending_context: list[Message] = []

        def answer(value) -> None:
            holder["value"] = str(value)

        def ask_user(question: str) -> str:
            q = str(question)
            reply = channel.ask_user(q)
            emit(Step(depth=depth, index=current["index"],
                              interaction=Interaction(question=q, answer=reply)))
            return reply

        def ask_parent(question: str) -> str:
            q = str(question)
            reply = str(clarify_handler(q))
            emit(Step(depth=depth, index=current["index"],
                              interaction=Interaction(question=q, answer=reply,
                                                      kind="parent")))
            pending_context.append(Message(
                "user",
                f"[Clarification from the parent agent]\nYour question: {q}\n"
                f"Parent's answer: {reply}",
            ))
            return reply

        def _distill_messages(text: str, system: str | None) -> list[Message]:
            msgs = []
            if system:
                msgs.append(Message("system", system))
            msgs.append(Message("user", text))
            return msgs

        def _record_distillation(text: str, msgs: list[Message], resp) -> str:
            before = meter.total_tokens
            meter.record(resp.usage, sent_messages=msgs, response_text=resp.content)
            out = resp.content or ""
            emit(Step(
                depth=depth, index=current["index"],
                llm_call=LLMCall(prompt=text, response=out,
                                 tokens=meter.total_tokens - before),
            ))
            return out

        def llm(text: str, system: str | None = None) -> str:
            if meter.exhausted:
                return _BUDGET_NOTICE
            msgs = _distill_messages(text, system)
            resp = self.client.complete(msgs, model=self.model)
            return _record_distillation(text, msgs, resp)

        def llm_batched(texts, system: str | None = None) -> list[str]:
            texts = list(texts)
            if not texts:
                return []
            if meter.exhausted:
                return [_BUDGET_NOTICE] * len(texts)

            def _call(t: str):
                try:
                    msgs = _distill_messages(t, system)
                    return msgs, self.client.complete(msgs, model=self.model), None
                except Exception as e:  # one failure must not sink the batch
                    return _distill_messages(t, system), None, e

            workers = max(1, min(len(texts), self.max_parallel_calls))
            slots: list = [None] * len(texts)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_call, t): i for i, t in enumerate(texts)}
                for fut in as_completed(futs):
                    slots[futs[fut]] = fut.result()

            outs: list[str] = []
            for t, (msgs, resp, err) in zip(texts, slots):
                if err is not None:
                    out = f"[llm error: {err}]"
                    emit(Step(depth=depth, index=current["index"],
                                      llm_call=LLMCall(prompt=t, response=out, tokens=0)))
                else:
                    out = _record_distillation(t, msgs, resp)
                outs.append(out)
            return outs

        def spawn(subtask: str, scope=None) -> str:
            if meter.exhausted:
                return _BUDGET_NOTICE
            sub_caps = (
                [c.subset(scope) for c in capabilities] if scope is not None
                else capabilities
            )
            child_id = f"{node_id}.{spawned['n']}"
            spawned["n"] += 1
            emit(Step(depth=depth, index=current["index"], spawn=subtask,
                      child_id=child_id))

            def clarify(child_question: str) -> str:
                ctx = list(messages) + [Message(
                    "user",
                    "A sub-task you delegated needs clarification before it can "
                    f'proceed:\n\n  "{child_question}"\n\nAnswer concisely from '
                    "what you already know about the overall goal. If you don't "
                    "know either, say so briefly.",
                )]
                resp = self.client.complete(ctx, model=self.model)
                meter.record(resp.usage, sent_messages=ctx, response_text=resp.content)
                return resp.content or ""

            # At the depth cap, don't degrade to a context-/tool-less flat call:
            # run a real sub-agent with the SAME capabilities that simply may not
            # recurse further. This keeps leaf work fully capable.
            at_cap = depth + 1 > self.max_depth
            return self.run(
                subtask,
                sub_caps,
                instructions=instructions,
                depth=depth if at_cap else depth + 1,
                node_id=child_id,
                channel=channel,
                meter=meter,
                clarify_handler=clarify,
                enable_planning=enable_planning,
                planning_root_only=planning_root_only,
                allow_spawn=(not at_cap) and (depth + 1 < self.max_depth),
            )

        # Assemble the REPL namespace: core helpers + every capability's names.
        # `spawn` is withheld when recursion is disabled (worker agents).
        names = {"llm": llm, "llm_batched": llm_batched, "answer": answer}
        if allow_spawn:
            names["spawn"] = spawn
        cap_ctx = CapabilityContext(depth=depth, workspace=self.workspace)
        for cap in capabilities:
            names.update(cap.bind(cap_ctx))

        system_prompt = build_system_prompt(
            capabilities, instructions=instructions, extra=self.extra_instructions,
            allow_spawn=allow_spawn,
        )
        planning_here = enable_planning and (not planning_root_only or depth == 0)
        if planning_here:
            system_prompt += PLANNING_PROMPT

            def on_todo_change(items: list[TodoItem]) -> None:
                emit(Step(depth=depth, index=current["index"], todo=items))

            names["todo"] = TodoList(on_todo_change)
        if self.enable_interaction:
            if not self.interaction_root_only or depth == 0:
                system_prompt += ASK_USER_PROMPT
                names["ask_user"] = ask_user
            if clarify_handler is not None:
                system_prompt += ASK_PARENT_PROMPT
                names["ask_parent"] = ask_parent
        env.inject(**names)

        messages: list[Message] = [Message("system", system_prompt)]
        messages.append(Message("user", task))

        for i in range(self.max_steps):
            current["index"] = i
            if meter.exhausted:
                emit(Step(depth=depth, index=i,
                                  note="budget exhausted — stopping before further model calls"))
                break
            resp = self.client.complete(messages, tools=[PYTHON_TOOL], model=self.model)
            context_tokens, context_estimated = meter.record(
                resp.usage, sent_messages=messages, response_text=resp.content
            )
            emit(Step(depth=depth, index=i,
                              usage=meter.snapshot(context_tokens, context_estimated)))
            if resp.content and resp.tool_calls:
                emit(Step(depth=depth, index=i, message=resp.content))
            step_start = len(messages)
            messages.append(
                Message("assistant", content=resp.content, tool_calls=resp.tool_calls)
            )

            if not resp.tool_calls:
                final = resp.content or holder.get("value", "")
                emit(Step(depth=depth, index=i, final=final))
                return _finish(final)

            for tc in resp.tool_calls:
                code = tc.arguments.get("code", "")
                emit(Step(depth=depth, index=i, code=code))
                result = env.execute(code)
                output = result.stdout
                if result.error:
                    output = (output + "\n" if output else "") + result.error
                output = output.strip() or "(no output)"
                output = truncate_output(output, self.max_output_chars)
                emit(Step(depth=depth, index=i, output=output))
                messages.append(Message("tool", content=output, tool_call_id=tc.id))

            if pending_context:
                messages.extend(pending_context)
                pending_context.clear()

            if "value" in holder:
                emit(Step(depth=depth, index=i, final=holder["value"]))
                return _finish(holder["value"])

            if self.compactor is not None:
                added = estimate_message_tokens(messages[step_start:])
                projected = context_tokens + added
                if self.compactor.should_compact(projected):
                    before_n = len(messages)
                    est_before = estimate_message_tokens(messages) or 1
                    scale = projected / est_before
                    messages, summarized = self.compactor.compact(
                        messages,
                        client=self.client,
                        model=self.model,
                        meter=meter,
                        state_description=env.describe_state(),
                    )
                    if summarized:
                        emit(Step(
                            depth=depth, index=i,
                            compaction=CompactionEvent(
                                tokens_before=round(projected),
                                tokens_after=round(estimate_message_tokens(messages) * scale),
                                messages_before=before_n,
                                messages_after=len(messages),
                                summarized_messages=summarized,
                            ),
                        ))

        if "value" in holder:
            return _finish(holder["value"])
        reason = "budget exhausted" if meter.exhausted else "reached max steps"

        if not meter.exhausted:
            synth = list(messages) + [Message("user", _PARTIAL_SYNTHESIS_PROMPT)]
            try:
                resp = self.client.complete(synth, model=self.model)
            except Exception:
                resp = None
            if resp is not None:
                meter.record(resp.usage, sent_messages=synth, response_text=resp.content)
                partial = (resp.content or "").strip()
                if partial:
                    emit(Step(depth=depth, index=current["index"], final=partial))
                    return _finish(partial)
        return _finish(f"Stopped: {reason} without a final answer.")
