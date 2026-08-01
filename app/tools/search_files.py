"""Search for text in files (like a simple grep)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List

from app.tools.base import BaseTool, ToolResult


_IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".next", "dist", "build", ".mypy_cache", ".pytest_cache",
    "target", "out",
}

_IGNORE_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico",
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".mp3", ".mp4", ".mov", ".webm", ".ogg",
    ".pyc", ".pyo", ".class", ".o", ".a", ".so", ".dll", ".exe",
    ".lock", ".sum",
}


def _iter_files(root: Path, glob: str):
    pattern = glob or "*"
    for path in root.rglob(pattern):
        if not path.is_file():
            continue
        # Skip ignored dirs
        if any(part in _IGNORE_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in _IGNORE_EXTS:
            continue
        yield path


class SearchFilesTool(BaseTool):
    name = "search_files"
    description = (
        "Search for a regex pattern in files under the workspace. Returns matching "
        "lines with file:line:content. Use glob to filter files (e.g. '*.py'). "
        "Case-insensitive by default. Use case_sensitive=true to override."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search for."},
            "glob": {
                "type": "string",
                "description": "File glob to filter (e.g. '*.py'). Default '*'.",
                "default": "*",
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "Case-sensitive match. Default false.",
                "default": False,
            },
            "max_results": {
                "type": "integer",
                "description": "Cap on matches. Default 200.",
                "default": 200,
            },
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    async def run(self, **kwargs: Any) -> ToolResult:
        pattern = kwargs.get("pattern", "")
        glob = kwargs.get("glob", "*")
        case_sensitive = bool(kwargs.get("case_sensitive", False))
        max_results = int(kwargs.get("max_results", 200))

        if not pattern:
            return ToolResult("error: pattern is empty", error=True)
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            rx = re.compile(pattern, flags)
        except re.error as e:
            return ToolResult(f"error: invalid regex: {e}", error=True)

        matches: List[str] = []
        files_scanned = 0
        truncated = False

        try:
            for path in _iter_files(self.ctx.workdir, glob):
                files_scanned += 1
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except (OSError, UnicodeError):
                    continue
                rel = path.relative_to(self.ctx.workdir)
                for i, line in enumerate(text.splitlines(), start=1):
                    if rx.search(line):
                        matches.append(f"{rel}:{i}:{line.rstrip()}")
                        if len(matches) >= max_results:
                            truncated = True
                            break
                if truncated:
                    break
        except Exception as e:
            return ToolResult(f"error during search: {e}", error=True)

        if not matches:
            return ToolResult(
                f"[no matches] (scanned {files_scanned} files)",
                metadata={"count": 0, "files_scanned": files_scanned},
            )
        out = "\n".join(matches)
        if truncated:
            out += f"\n... [truncated, hit max_results={max_results}]"
        out = self.cap(out, 16000)
        return ToolResult(
            out,
            metadata={
                "count": len(matches),
                "files_scanned": files_scanned,
                "truncated": truncated,
            },
        )
