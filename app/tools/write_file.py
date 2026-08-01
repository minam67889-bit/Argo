"""Write (create or overwrite) a file."""
from __future__ import annotations

from typing import Any

from app.tools.base import BaseTool, ToolResult


class WriteFileTool(BaseTool):
    name = "write_file"
    description = (
        "Create or overwrite a file. Parent directories are created automatically. "
        "Use this for new files or when rewriting a file wholesale. For small "
        "changes, prefer edit_file."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to workspace."},
            "content": {"type": "string", "description": "Full file content."},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    async def run(self, **kwargs: Any) -> ToolResult:
        path = kwargs.get("path", "")
        content = kwargs.get("content", "")

        if not path:
            return ToolResult("error: path is empty", error=True)
        if not isinstance(content, str):
            content = str(content)

        try:
            p = self.resolve_path(path)
        except (PermissionError, ValueError) as e:
            return ToolResult(f"error: {e}", error=True)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            size = p.stat().st_size
            return ToolResult(
                f"[written] {path} ({size} bytes)",
                error=False,
                metadata={"path": path, "size": size},
            )
        except Exception as e:
            return ToolResult(f"error writing {path}: {e}", error=True)
