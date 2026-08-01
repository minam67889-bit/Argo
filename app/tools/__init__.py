"""Tool registry. Adding a tool: subclass BaseTool, register, done."""
from __future__ import annotations

from typing import Dict, Type

from app.tools.base import BaseTool, ToolContext, ToolResult
from app.tools.bash import BashTool
from app.tools.read_file import ReadFileTool
from app.tools.write_file import WriteFileTool
from app.tools.edit_file import EditFileTool
from app.tools.list_dir import ListDirTool
from app.tools.search_files import SearchFilesTool

# Public registry. Order matters for UI display.
ALL_TOOLS: Dict[str, Type[BaseTool]] = {
    "bash": BashTool,
    "read_file": ReadFileTool,
    "write_file": WriteFileTool,
    "edit_file": EditFileTool,
    "list_dir": ListDirTool,
    "search_files": SearchFilesTool,
}

# Default tools for the agent loop (all of them, in this order).
DEFAULT_TOOL_NAMES = list(ALL_TOOLS.keys())


def make_context(workdir, auto_approve: bool = False, bash_timeout: int = 120) -> ToolContext:
    """Build a ToolContext bound to a workspace."""
    return ToolContext(
        workdir=workdir,
        auto_approve=auto_approve,
        bash_timeout=bash_timeout,
    )


def get_tool(name: str, ctx: ToolContext) -> BaseTool:
    """Get a tool instance by name, configured for the given context."""
    if name not in ALL_TOOLS:
        raise KeyError(f"Unknown tool: {name}")
    return ALL_TOOLS[name](ctx)


__all__ = [
    "BaseTool",
    "ToolContext",
    "ToolResult",
    "ALL_TOOLS",
    "DEFAULT_TOOL_NAMES",
    "make_context",
    "get_tool",
]
