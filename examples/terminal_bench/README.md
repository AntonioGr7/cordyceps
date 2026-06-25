# Cordyceps on Terminal-Bench

Run the Cordyceps agent against [Terminal-Bench](https://www.tbench.ai/) tasks.

## Setup

1. **Docker** must be installed and running (tasks execute in containers).
2. Install the harness and Cordyceps:
   ```bash
   pip install terminal-bench
   pip install -e .            # from the repo root
   ```
3. Configure `.env` for your provider (see the main README) — `anthropic` or
   `openai`. The adapter reads it via `Settings.from_env()`.

## Run a few tasks

From the repo root:

```bash
# 2 tasks from the default dataset
tb run --agent-import-path examples.terminal_bench.cordyceps_tb_agent:CordycepsAgent -n 2

# or a specific task by id
tb run --agent-import-path examples.terminal_bench.cordyceps_tb_agent:CordycepsAgent \
       --task-id hello-world
```

`tb tasks list` shows available task ids. Results (pass/fail per task, logs,
recordings) are written under `runs/` by the harness.

## How it works

`cordyceps_tb_agent.py` defines:

- **`TmuxShellCapability`** — a Cordyceps capability whose `sh(command)` sends the
  command into the benchmark's `TmuxSession` (the task container) and parses its
  output + exit code back out via line-unique sentinels.
- **`CordycepsAgent(BaseAgent)`** — implements `name()` + `perform_task(...)`. It
  builds an engine from your `.env`, mounts the tmux shell, runs the task, and
  returns token counts as an `AgentResult`.

Cordyceps' reasoning runs on the host; only the *shell* is redirected into the
container. `spawn()` still works — sub-agents share the same tmux session (same
container), so decomposition operates on one terminal.

## Tuning notes

- The output parser (`_extract`) assumes commands finish within the tmux
  `block=True` window and that markers land on their own lines. Very long-running
  or TUI-heavy commands may need a larger `default_timeout` or a different capture
  strategy.
- Set `CORDYCEPS_MAX_TOTAL_TOKENS` / `CORDYCEPS_MAX_STEPS` to bound cost per task.
- `CORDYCEPS_EFFORT=high` (Anthropic) tends to help on harder terminal tasks.
