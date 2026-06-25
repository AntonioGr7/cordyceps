"""Run a standalone Cordyceps Agent on a small real task.

    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/basic_agent.py

The agent gets shell + filesystem capabilities and a workspace, then writes and
runs a small program — all via code it authors in its REPL.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os

from cordyceps import Agent, RunLogger
from cordyceps.capabilities import FileSystemCapability, ShellCapability


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSION_DIR = os.path.join(REPO, "data", "session")


def main() -> None:
    workdir = os.path.join(SESSION_DIR, "work")
    os.makedirs(workdir, exist_ok=True)

    agent = Agent(
        role="builder",
        description=(
            "You implement small, well-scoped programming tasks by writing files "
            "and running them. Verify your work actually runs before answering."
        ),
        capabilities=[
            ShellCapability(cwd=workdir),
            FileSystemCapability(root=workdir),
        ],
        max_total_tokens=200_000,
    )

    # RunLogger streams the live split to stderr AND records a full JSONL log.
    log = RunLogger(jsonl_path=os.path.join(SESSION_DIR, "run.jsonl"), live=True)
    answer = agent.solve(
        "Write fib.py that prints the first 10 Fibonacci numbers, run it, and "
        "report the output.",
        on_event=log,
    )
    log.close()

    print("\n=== DECOMPOSITION TREE ===")
    print(log.render_tree())
    print("\n=== SUMMARY ===", log.summary())
    print("\n=== ANSWER ===")
    print(answer)


if __name__ == "__main__":
    main()
