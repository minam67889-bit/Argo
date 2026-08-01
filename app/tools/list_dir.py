"""List directory contents."""
from __future__ import annotations

import os
from typing import Any

from app.tools.base import BaseTool, ToolResult


_IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".next", "dist", "build"}


class ListDirTool(BaseTool):
    name = "list_dir"
    description = (
        "List the contents of a directory. With recursive=true, walks subdirectories "
        "(skipping build/dependency dirs). With recursive=false, lists one level."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory to list. Defaults to workspace root.",
                "default": ".",
            },
            "recursive": {
                "type": "boolean",
                "description": "Walk subdirectories.",
                "default": False,
            },
        },
        "additionalProperties": False,
    }

    async def run(self, **kwargs: Any) -> ToolResult:
        path = kwargs.get("path", ".") or "."
        recursive = bool(kwargs.get("recursive", False))

        try:
            p = self.resolve_path(path)
        except (PermissionError, ValueError) as e:
            return ToolResult(f"error: {e}", error=True)
        if not p.exists():
            return ToolResult(f"error: directory not found: {path}", error=True)
        if not p.is_dir():
            return ToolResult(f"error: not a directory: {path}", error=True)

        try:
            if recursive:
                rows = []
                for root, dirs, files in os.walk(p):
                    dirs[:] = [d for d in dirs if d not in _IGNORE_DIRS]
                    rel_root = os.path.relpath(root, p) or "."
                    if rel_root == ".":
                        for f in sorted(files):
                            rows.append(f)
                    else:
                        for f in sorted(files):
                            rows.append(os.path.join(rel_root, f))
                if not rows:
                    return ToolResult("[empty]", metadata={"count": 0})
                rows = rows[:500]
                return ToolResult(
                    "\n".join(rows),
                    metadata={"count": len(rows), "truncated": len(rows) >= 500},
                )
            else:
                entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
                rows = []
                for x in entries:
                    suffix = "/" if x.is_dir() else ""
                    rows.append(x.name + suffix)
                if not rows:
                    return ToolResult("[empty]", metadata={"count": 0})
                return ToolResult("\n".join(rows), metadata={"count": len(rows)})
        except Exception as e:
            return ToolResult(f"error: {e}", error=True)
