"""Edit a file: find old_text and replace with new_text. Fuzzy matching on whitespace."""
from __future__ import annotations

import re
from typing import Any

from app.tools.base import BaseTool, ToolResult


def _normalize(s: str) -> str:
    """Collapse all whitespace runs to a single space, strip ends."""
    return re.sub(r"\s+", " ", s).strip()


def _fuzzy_replace(text: str, old: str, new: str) -> tuple[str, bool]:
    """Try exact match first, then fuzzy whitespace-tolerant match.

    Returns (new_text, matched).
    """
    if not old:
        return text, False

    # 1) Exact match
    if old in text:
        return text.replace(old, new, 1), True

    # 2) Line-based fuzzy: walk through windows of lines, look for normalized match.
    nold = _normalize(old)
    lines = text.splitlines(keepends=True)
    for i in range(len(lines)):
        for j in range(i + 1, min(i + 200, len(lines)) + 1):
            block = "".join(lines[i:j])
            if _normalize(block) == nold:
                head = "".join(lines[:i])
                tail = "".join(lines[j:])
                return head + new + tail, True
    return text, False


class EditFileTool(BaseTool):
    name = "edit_file"
    description = (
        "Edit a file by replacing the first occurrence of old_text with new_text. "
        "Whitespace-tolerant: collapses all whitespace (spaces, tabs, newlines) "
        "for the match, so minor formatting differences still match. "
        "If old_text appears multiple times, only the first is replaced."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to workspace."},
            "old_text": {"type": "string", "description": "Text to find."},
            "new_text": {"type": "string", "description": "Replacement text."},
            "replace_all": {
                "type": "boolean",
                "description": "If true, replace every occurrence. Default false.",
                "default": False,
            },
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    }

    async def run(self, **kwargs: Any) -> ToolResult:
        path = kwargs.get("path", "")
        old = kwargs.get("old_text", "")
        new = kwargs.get("new_text", "")
        replace_all = bool(kwargs.get("replace_all", False))

        if not path:
            return ToolResult("error: path is empty", error=True)
        if not old:
            return ToolResult("error: old_text is empty", error=True)

        try:
            p = self.resolve_path(path)
        except (PermissionError, ValueError) as e:
            return ToolResult(f"error: {e}", error=True)
        if not p.exists():
            return ToolResult(f"error: file not found: {path}", error=True)

        try:
            text = p.read_text(encoding="utf-8")
        except Exception as e:
            return ToolResult(f"error reading {path}: {e}", error=True)

        if replace_all and old in text:
            count = text.count(old)
            new_text = text.replace(old, new)
            p.write_text(new_text, encoding="utf-8")
            return ToolResult(
                f"[edited] {path} ({count} replacement{'s' if count != 1 else ''})",
                metadata={"path": path, "count": count},
            )

        new_text, matched = _fuzzy_replace(text, old, new)
        if not matched:
            return ToolResult(
                f"error: old_text not found in {path} (tried exact and fuzzy)",
                error=True,
            )
        p.write_text(new_text, encoding="utf-8")
        return ToolResult(f"[edited] {path}", metadata={"path": path})
