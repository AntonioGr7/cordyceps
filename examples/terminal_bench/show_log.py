"""Pretty-print a saved Cordyceps run log (cordyceps_run.jsonl).

    # most recent task log in the latest run:
    python examples/terminal_bench/show_log.py
    # or a specific file:
    python examples/terminal_bench/show_log.py runs/<id>/<task>/<trial>/agent-logs/cordyceps_run.jsonl

Shows each step as a timeline (» code / « output / ↳ split / ✓ final), the same
view as RunLogger's verbose live mode but replayed from disk.
"""

from __future__ import annotations

import glob
import json
import os
import sys


def latest_log() -> str | None:
    hits = glob.glob("runs/**/agent-logs/cordyceps_run.jsonl", recursive=True)
    return max(hits, key=os.path.getmtime) if hits else None


def render(path: str, value_limit: int = 600) -> None:
    print(f"# {path}\n")
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        ind = "  " * ((e.get("node_id") or "0").count("."))
        if e.get("spawn"):
            print(f"{ind}↳ split [{e.get('child_id')}] {e['spawn']}")
        if e.get("code"):
            body = e["code"].rstrip()
            print(f"{ind}» " + body.replace("\n", "\n" + ind + "  "))
        if e.get("output") is not None:
            out = e["output"]
            if len(out) > value_limit:
                out = out[:value_limit] + f"… (+{len(out) - value_limit} chars)"
            print(f"{ind}« " + out.replace("\n", "\n" + ind + "  "))
        if e.get("note"):
            print(f"{ind}! {e['note']}")
        if e.get("final") is not None:
            print(f"{ind}✓ → {e['final']}")


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else latest_log()
    if not path or not os.path.exists(path):
        print("No log found. Pass a path to a cordyceps_run.jsonl, or run a task first.")
        sys.exit(1)
    render(path)


if __name__ == "__main__":
    main()
