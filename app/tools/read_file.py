"""Read a file. Supports line ranges for large files."""
from __future__ import annotations

from typing import Any

from app.tools.base import BaseTool, ToolResult


class ReadFileTool(BaseTool):
    name = "read_file"
    description = (
        "Read a file's content. For large files, use start_line/end_line to read "
        "a specific range. Returns UTF-8 text (invalid bytes are replaced)."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to workspace."},
            "start_line": {
                "type": "integer",
                "description": "0-indexed start line (inclusive). Optional.",
            },
            "end_line": {
                "type": "integer",
                "description": "0-indexed end line (exclusive). Optional.",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    async def run(self, **kwargs: Any) -> ToolResult:
        path = kwargs.get("path", "")
        start = kwargs.get("start_line")
        end = kwargs.get("end_line")

        try:
            p = self.resolve_path(path)
        except (PermissionError, ValueError) as e:
            return ToolResult(f"error: {e}", error=True)
        if not p.exists():
            return ToolResult(f"error: file not found: {path}", error=True)
        if p.is_dir():
            return ToolResult(f"error: is a directory (use list_dir): {path}", error=True)

        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return ToolResult(f"error reading {path}: {e}", error=True)

        if start is not None or end is not None:
            lines = content.splitlines()
            s = max(0, int(start) if start is not None else 0)
            e = min(len(lines), int(end) if end is not None else len(lines))
            content = "\n".join(lines[s:e])
            content = f"[lines {s}-{e} of {len(lines)}]\n" + content

        out = self.cap(content, 16000)
        return ToolResult(out, error=False, metadata={"path": path, "size": p.stat().st_size})
