"""Bash tool — run a shell command inside the workdir."""
from __future__ import annotations

import asyncio
import shlex
from typing import Any

from app.tools.base import BaseTool, ToolContext, ToolResult


# Commands we always refuse, no matter what. Defense in depth.
_DENY_PATTERNS = (
    "rm -rf /",
    "rm -rf /*",
    "rm -rf ~",
    ":(){:|:&};:",  # fork bomb
    "dd if=",
    "mkfs.",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "init 0",
    "init 6",
    "> /dev/sd",
)


class BashTool(BaseTool):
    name = "bash"
    description = (
        "Run a shell command. Use for builds, tests, git, grep, find, "
        "pip install, unzip, etc. Working directory is the workspace. "
        "Output is captured (stdout+stderr) and returned. Long-running commands "
        "are killed after the configured timeout."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "cmd": {
                "type": "string",
                "description": "The shell command to run.",
            },
        },
        "required": ["cmd"],
        "additionalProperties": False,
    }

    def _is_denied(self, cmd: str) -> bool:
        c = cmd.strip().lower()
        return any(p in c for p in _DENY_PATTERNS)

    async def run(self, **kwargs: Any) -> ToolResult:
        cmd = kwargs.get("cmd", "")
        if not isinstance(cmd, str) or not cmd.strip():
            return ToolResult("error: empty command", error=True)

        if self._is_denied(cmd):
            return ToolResult(
                f"error: command denied by safety filter: {cmd[:80]!r}",
                error=True,
            )

        # Optional user approval (CLI mode). The API doesn't pass a hook, so
        # auto_approve is what controls this in server mode.
        if not self.ctx.auto_approve and self.ctx.on_bash_request is not None:
            approved = await self.ctx.on_bash_request(cmd)
            if not approved:
                return ToolResult("[skipped by user]", error=False)

        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                cwd=str(self.ctx.workdir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=self.ctx.bash_timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return ToolResult(
                    f"[timeout after {self.ctx.bash_timeout}s] {cmd}",
                    error=True,
                    metadata={"exit_code": -1, "timed_out": True},
                )

            stdout = stdout_b.decode("utf-8", errors="replace")
            stderr = stderr_b.decode("utf-8", errors="replace")
            out = stdout
            if stderr:
                out += "\n[stderr]\n" + stderr
            out = out.strip()
            if not out:
                out = "[no output]"
            out = self.cap(out, 12000)

            exit_code = proc.returncode or 0
            if exit_code != 0:
                out += f"\n[exit {exit_code}]"
            return ToolResult(
                out,
                error=(exit_code != 0),
                metadata={"exit_code": exit_code, "command": cmd[:200]},
            )
        except Exception as e:
            return ToolResult(f"error: {type(e).__name__}: {e}", error=True)
