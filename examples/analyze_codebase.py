"""A complex, decomposition-heavy test: have the swarm analyze a codebase.

The agent is pointed at the Cordyceps source tree and asked to review it module
by module. Because the modules are independent, this naturally fans out: the root
agent spawns a sub-agent per module, each reads + summarizes its files, and the
root synthesizes the results into one architecture review.

    # .env configured for your provider (anthropic or openai), then:
    python examples/analyze_codebase.py

Watch stderr for the live split tree; the artifact and full event log land under
data/session/ (gitignored).
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import os

from cordyceps import Agent, RunLogger
from cordyceps.capabilities import FileSystemCapability, ShellCapability

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSION_DIR = os.path.join(REPO, "data", "session")

TASK = """\
Review the Python codebase under src/cordyceps/ and produce an architecture review.

Work like this:
  1. First, list the modules/sub-packages under src/cordyceps/ (e.g. engine,
     capabilities, llm, plus the top-level modules agent.py, capability.py,
     config.py, observability.py).
  2. For EACH module that is substantial enough to warrant it, spawn a sub-agent
     whose subtask is: "Read every .py file under <module path> and report its
     responsibility, the key classes/functions, and what other cordyceps modules
     it depends on." Spawn them as independent sub-problems.
  3. Collect the sub-agents' summaries and synthesize ONE coherent document:
       - a 1-paragraph overview of what Cordyceps is and how the pieces fit,
       - a section per module (responsibility + key types + dependencies),
       - a short "data & control flow" description of how a task is executed.
  4. Write that document to data/session/ARCHITECTURE_REVIEW.md.
  5. answer() with a 3-sentence executive summary plus the path you wrote.
"""


def main() -> None:
    agent = Agent(
        role="architecture-reviewer",
        description=(
            "You analyze codebases. You decompose the work by module, delegating "
            "each module's analysis to a sub-agent via spawn(), then synthesize "
            "their findings. Read files before describing them; never invent APIs."
        ),
        capabilities=[
            ShellCapability(cwd=REPO),
            FileSystemCapability(root=REPO),
        ],
        max_total_tokens=600_000,
    )

    os.makedirs(SESSION_DIR, exist_ok=True)
    log = RunLogger(jsonl_path=os.path.join(SESSION_DIR, "analyze_run.jsonl"), live=True)
    answer = agent.solve(TASK, on_event=log)
    log.close()

    print("\n=== DECOMPOSITION TREE ===")
    print(log.render_tree())
    print("\n=== SUMMARY ===", log.summary())
    print("\n=== ANSWER ===")
    print(answer)


if __name__ == "__main__":
    main()
