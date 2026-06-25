"""DataStore capability — a shared blackboard for passing data between agents.

The RLM premise is that raw data never enters the model's context. Input datasets
and intermediate results therefore live as Python objects in a store, and agents
exchange them by *key* (a small handle) rather than by value. The model reads a
key into a REPL variable, works on it, and writes a result back under a new key —
the bytes never touch the transcript.

Because the engine passes the SAME capability instance down the whole spawn tree
(`subset` is identity), one DataStore is automatically shared by the root agent
and every sub-agent it spawns. That makes it the substrate for DAG dataflow:
edges carry keys, the store carries the bytes. Seed it before the run to feed
input data in; read it after the run to collect the output.

    store = DataStore({"sales": df})
    agent = Agent(role="analyst", description="...", capabilities=[store])
    agent.solve("Analyze data('sales') and put the summary table under 'summary'.")
    result = store.get("summary")
"""

from __future__ import annotations

import threading
from typing import Any

from ..capability import BaseCapability, CapabilityContext

_SURFACE = """  data(key) -> object               Read a shared data object by key (e.g. a
                                    DataFrame, list, or dict). Raw data lives
                                    here, NOT in your context — pull it into a
                                    variable and work on it.
  put_data(key, value) -> str       Store a result object under `key` for the
                                    parent or sibling agents to read. Pass the
                                    KEY forward (e.g. via spawn), not the value.
  data_keys() -> list[str]          List the keys currently in the shared store."""


class DataStore(BaseCapability):
    name = "datastore"

    def __init__(self, initial: dict[str, Any] | None = None):
        self._data: dict[str, Any] = dict(initial or {})
        self._lock = threading.Lock()  # siblings may run concurrently

    # -- host-side access (seed input / collect output) -----------------
    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def get(self, key: str) -> Any:
        with self._lock:
            return self._data[key]

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._data)

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)

    # -- capability surface (REPL names) --------------------------------
    def bind(self, ctx: CapabilityContext) -> dict[str, Any]:
        def data(key: str) -> Any:
            with self._lock:
                if key not in self._data:
                    raise KeyError(
                        f"no data under key {key!r}; available: {list(self._data)}"
                    )
                return self._data[key]

        def put_data(key: str, value: Any) -> str:
            with self._lock:
                self._data[key] = value
            return f"stored data under key {key!r}"

        def data_keys() -> list[str]:
            with self._lock:
                return list(self._data)

        return {"data": data, "put_data": put_data, "data_keys": data_keys}

    def surface(self) -> str:
        return _SURFACE
