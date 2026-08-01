#!/usr/bin/env python3
"""Argo CLI: run the agent from your terminal.

Usage:
  python -m cli.agent "fix the bug in foo.py"
  python -m cli.agent --workdir /path/to/project --auto
  python -m cli.agent    # interactive REPL
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent import AgentLoop
from app.core.config import settings
from app.core.llm import LLMClient
from app.tools.base import ToolContext

# Add CWD to path so 'app' is importable when running as a script too
if not (Path.cwd() / "app").exists() and (Path(__file__).parent.parent / "app").exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))


# ---- Color helpers ----
def _c(code: int) -> str:
    return f"\033[{code}m"

def _wrap(s: str, code: int) -> str:
    if not sys.stdout.isatty():
        return s
    return f"{_c(code)}{s}{_c(0)}"

def green(s): return _wrap(s, 32)
def yellow(s): return _wrap(s, 33)
def cyan(s): return _wrap(s, 36)
def red(s): return _wrap(s, 31)
def dim(s): return _wrap(s, 90)
def bold(s): return _wrap(s, 1)


async def ask_bash(cmd: str) -> bool:
    """Ask the user to approve a bash command (CLI mode)."""
    print()
    print(yellow(f"  ? اجرا بشه؟ {cmd}"))
    try:
        ans = input(yellow("    [Y/n] ")).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return ans in ("", "y", "yes")


async def run_interactive(args):
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    if not settings.LLM_API_KEY:
        print(red("LLM_API_KEY تنظیم نشده."))
        print("مثال:")
        print("  export LLM_API_KEY='sk-or-...'")
        sys.exit(1)

    print(bold(green("Argo")) + f" | مدل: {cyan(settings.LLM_MODEL)} | پوشه: {dim(str(workdir))}")
    print(dim(" — برای خروج: exit | خالی = ادامه روی همون context —\n"))

    client = LLMClient()
    auto_approve = args.auto
    on_bash = None if auto_approve else ask_bash

    loop = AgentLoop(
        llm=client,
        workdir=workdir,
        mode="agent",
        auto_approve=auto_approve,
        on_bash_request=on_bash,
    )

    if args.task:
        # One-shot
        await run_session(loop, " ".join(args.task))
        return

    # Interactive
    while True:
        try:
            line = input(cyan("you» ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nخداحافظ.")
            break
        if not line:
            continue
        if line.lower() in ("exit", "quit", "/exit", "/q"):
            break
        if line.lower() == "/reset":
            loop.reset()
            print(dim("(context پاک شد)"))
            continue
        if line.lower() == "/mode chat":
            loop.mode = "chat"
            print(dim("(mode = chat)"))
            continue
        if line.lower() == "/mode agent":
            loop.mode = "agent"
            print(dim("(mode = agent)"))
            continue
        await run_session(loop, line)


async def run_session(loop, user_input):
    try:
        async for ev in loop.run(user_input):
            if ev.type == "text":
                print(ev.data.get("content", ""), end="", flush=True)
            elif ev.type == "reasoning":
                # optional: show reasoning in dim
                print(dim(ev.data.get("content", "")), end="", flush=True)
            elif ev.type == "step":
                print(dim(f"\n── گام {ev.data['step']}/{ev.data['max']} ──"))
            elif ev.type == "tool_call":
                name = ev.data.get("name", "?")
                args = ev.data.get("arguments", {})
                preview = str(args)[:140]
                print(yellow(f"\n🔧 {name}") + f" {preview}")
            elif ev.type == "tool_result":
                err = ev.data.get("error", False)
                out = ev.data.get("output", "")
                # Truncate for display
                if len(out) > 1500:
                    out = out[:1500] + dim(f"\n... [+{len(out)-1500} chars]")
                color = red if err else dim
                for line in out.splitlines()[:30]:
                    print(color(" │ " + line))
            elif ev.type == "error":
                print(red(f"\n✗ {ev.data.get('message', 'error')}"))
            elif ev.type == "done":
                print()
                print(dim(
                    f"── پایان: گام {ev.data.get('steps')}, "
                    f"{ev.data.get('elapsed', 0)}s, "
                    f"{ev.data.get('tokens', 0)} tokens ──"
                ))
                if ev.data.get("max_steps_reached"):
                    print(yellow("⚠ به حداکثر گام رسید."))
                print()
    except KeyboardInterrupt:
        print(red("\n(توقف با Ctrl+C)"))


def main():
    ap = argparse.ArgumentParser(description="Argo: ایجنت کدنویسی در ترمینال")
    ap.add_argument("task", nargs="*", help="تسک (اگه خالی، حالت تعاملی)")
    ap.add_argument("--workdir", "-d", default=str(settings.ARGO_WORKSPACE), help="پوشه‌ی کاری")
    ap.add_argument("--auto", action="store_true", help="بدون تأیید دستورات bash")
    ap.add_argument("--chat", action="store_true", help="حالت چت (بدون ابزار)")
    args = ap.parse_args()

    if args.chat:
        os.environ["AGENT_MODE"] = "chat"

    asyncio.run(run_interactive(args))


if __name__ == "__main__":
    main()
