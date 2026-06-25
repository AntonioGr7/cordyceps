"""Run the Cordyceps Swarm — the recursive problem-decomposition orchestrator.

    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/swarm_decompose.py

Unlike a single Agent, the Swarm's job is to SPLIT: a decomposer agent breaks the
problem into sub-problems with dependency edges, independent sub-problems run in
parallel, dependent ones pipe their output forward, and each sub-problem is itself
re-decomposed (lazily, in real time) only if it is still complex. Worker agents at
the leaves do the actual work with shell + filesystem capabilities.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os

from cordyceps import RunLogger, Swarm
from cordyceps.capabilities import FileSystemCapability, ShellCapability

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSION_DIR = os.path.join(REPO, "data", "session")


def main() -> None:
    workdir = os.path.join(SESSION_DIR, "swarm_work")
    os.makedirs(workdir, exist_ok=True)

    swarm = Swarm(
        capabilities=[
            ShellCapability(cwd=workdir),
            FileSystemCapability(root=workdir),
        ],
        max_depth=3,        # how deep decomposition may recurse
        max_breadth=6,      # max sub-problems per split
        max_parallel=4,     # independent sub-problems run concurrently
        max_total_tokens=400_000,
    )

    log = RunLogger(jsonl_path=os.path.join(SESSION_DIR, "swarm.jsonl"), live=True)
    answer = swarm.solve(
        "Build a small Python CLI 'wc-lite' that counts lines, words, and chars "
        "of a file; write a couple of unit tests for it; run the tests; and "
        "report whether they pass.",
        on_event=log,
        return_result=True,
    )
    log.close()

    print("\n=== DECOMPOSITION TREE ===")
    print(log.render_tree())
    print("\n=== SUMMARY ===", log.summary())
    print(f"tokens={answer.tokens} calls={answer.calls}")
    print("\n=== ANSWER ===")
    print(answer.answer)


if __name__ == "__main__":
    main()
