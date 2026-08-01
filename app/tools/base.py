"""Base class for tools. Each tool:
- has a name + description + JSON schema
- takes a ToolContext (workdir, timeouts, flags)
- returns a ToolResult (string output, error flag, metadata)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ToolContext:
    """Runtime context passed to every tool invocation."""
    workdir: Path
    auto_approve: bool = False
    bash_timeout: int = 120
    # Hook for the API/CLI to ask user before running a tool. Optional.
    on_bash_request: Optional[Any] = None  # Callable[[str], Awaitable[bool]]


@dataclass
class ToolResult:
    """Standard return value for any tool."""
    output: str
    error: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
        }


class BaseTool:
    """Subclass this to add a new tool. Override name, description, schema, run()."""

    name: str = ""
    description: str = ""
    # JSON schema describing the arguments the tool accepts.
    parameters_schema: Dict[str, Any] = {}

    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx

    def resolve_path(self, path: str) -> Path:
        """Resolve a user-supplied path inside the workdir. Raise on escape."""
        if not path:
            raise ValueError("path is empty")
        p = Path(path)
        if not p.is_absolute():
            p = self.ctx.workdir / p
        p = p.resolve()
        try:
            p.relative_to(self.ctx.workdir.resolve())
        except ValueError as e:
            raise PermissionError(f"Path escapes workspace: {path}") from e
        return p

    def to_openai_schema(self) -> Dict[str, Any]:
        """Return the OpenAI function-calling schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    async def run(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError

    # ---- Helpers for output capping ----
    @staticmethod
    def cap(text: str, limit: int, marker: str = "...[truncated]") -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n{marker} ({len(text) - limit} more chars)"

    def to_text(self, result: ToolResult) -> str:
        """Format a result for the model (just the output)."""
        return result.output
