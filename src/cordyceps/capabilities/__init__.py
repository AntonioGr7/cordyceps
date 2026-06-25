"""Built-in capabilities: the things an Agent can be given to touch the world."""

from __future__ import annotations

from .datastore import DataStore
from .filesystem import FileSystemCapability
from .shell import ShellCapability
from .tools import Tool, ToolRegistry

__all__ = [
    "ShellCapability",
    "FileSystemCapability",
    "DataStore",
    "ToolRegistry",
    "Tool",
]
