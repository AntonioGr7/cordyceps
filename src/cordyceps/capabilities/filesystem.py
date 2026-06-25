"""Filesystem capability — read, write, and edit real files.

`root`, if given, confines all paths beneath it (relative paths resolve against
it; paths escaping it are rejected). With no root, paths resolve against the cwd
and are unconfined — fine for a trusted local tool, not for untrusted input.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..capability import BaseCapability, CapabilityContext

_SURFACE = """  read_file(path) -> str            Read a text file and return its contents.
  write_file(path, text) -> str     Write text to a file (creating parent dirs);
                                    returns a short confirmation.
  edit_file(path, old, new) -> str  Replace the single occurrence of `old` with
                                    `new` in a file (errors if 0 or >1 matches).
  list_dir(path=".") -> list[str]   List directory entries.
  These operate on real files; hold large contents in a variable, don't print
  them wholesale."""


class FileSystemCapability(BaseCapability):
    name = "filesystem"

    def __init__(self, *, root: str | None = None):
        self.root = Path(root).resolve() if root else None

    def _resolve(self, path: str, base: Path) -> Path:
        p = Path(path)
        target = (p if p.is_absolute() else base / p).resolve()
        if self.root is not None and not _within(target, self.root):
            raise ValueError(f"path {path!r} escapes the filesystem root {self.root}")
        return target

    def bind(self, ctx: CapabilityContext) -> dict[str, Any]:
        base = self.root or (Path(ctx.workspace).resolve() if ctx.workspace else Path.cwd())

        def read_file(path: str) -> str:
            return self._resolve(path, base).read_text()

        def write_file(path: str, text: str) -> str:
            target = self._resolve(path, base)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
            return f"wrote {len(text)} chars to {target}"

        def edit_file(path: str, old: str, new: str) -> str:
            target = self._resolve(path, base)
            content = target.read_text()
            count = content.count(old)
            if count == 0:
                raise ValueError("old text not found")
            if count > 1:
                raise ValueError(f"old text matches {count} times; must be unique")
            target.write_text(content.replace(old, new, 1))
            return f"edited {target}"

        def list_dir(path: str = ".") -> list[str]:
            return sorted(os.listdir(self._resolve(path, base)))

        return {
            "read_file": read_file,
            "write_file": write_file,
            "edit_file": edit_file,
            "list_dir": list_dir,
        }

    def surface(self) -> str:
        return _SURFACE


def _within(target: Path, root: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except ValueError:
        return False
