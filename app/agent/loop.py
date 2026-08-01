"""Agent loop: drives the LLM in a ReAct-style cycle with streaming events."""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from app.agent.parser import (
    ToolCall,
    format_tool_result,
    parse_tool_calls,
    strip_tool_blocks,
)
from app.agent.prompts import build_agent_system_prompt, build_chat_system_prompt
from app.core.config import settings
from app.core.llm import LLMClient
from app.tools import DEFAULT_TOOL_NAMES, get_tool, make_context


@dataclass
class AgentEvent:
    """An event emitted by the agent loop. The API streams these as SSE."""
    type: str  # "text", "reasoning", "tool_call", "tool_result", "step", "done", "error", "usage"
    data: Dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        import json
        return f"data: {json.dumps({'type': self.type, **self.data}, ensure_ascii=False)}\n\n"


class AgentLoop:
    """Drives the conversation with the LLM, calling tools as needed.

    Yields AgentEvent objects. Caller decides how to serialize (SSE, log, etc).
    """

    def __init__(
        self,
        llm: LLMClient,
        workdir: Path,
        mode: str = "agent",  # "agent" or "chat"
        max_steps: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        auto_approve: Optional[bool] = None,
        bash_timeout: Optional[int] = None,
        tool_names: Optional[List[str]] = None,
        on_bash_request: Optional[Any] = None,
        model: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.workdir = workdir
        self.mode = mode
        self.max_steps = max_steps or settings.AGENT_MAX_STEPS
        self.max_tokens = max_tokens or settings.AGENT_MAX_TOKENS
        self.temperature = temperature if temperature is not None else settings.AGENT_TEMPERATURE
        self.tool_names = tool_names if tool_names is not None else (
            DEFAULT_TOOL_NAMES if mode == "agent" else []
        )
        self.auto_approve = auto_approve if auto_approve is not None else settings.AGENT_AUTO_APPROVE
        self.bash_timeout = bash_timeout or settings.AGENT_BASH_TIMEOUT
        self.on_bash_request = on_bash_request
        self.model = model

        # History (system + user + assistant + tool results)
        self.messages: List[Dict[str, Any]] = []
        self._init_system_message()

        # Loop detection: track recent (name, args-hash) tuples
        self._recent_calls: List[tuple] = []
        self._loop_threshold = settings.AGENT_LOOP_THRESHOLD

        # Counters
        self.steps = 0
        self.total_tokens = 0
        self.start_time = 0.0

    def _init_system_message(self) -> None:
        if self.mode == "agent":
            sys_prompt = build_agent_system_prompt(
                workdir=str(self.workdir),
                tool_names=self.tool_names,
            )
        else:
            sys_prompt = build_chat_system_prompt()
        self.messages.append({"role": "system", "content": sys_prompt})

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def reset(self) -> None:
        """Clear history (keep system message)."""
        sys_msg = self.messages[0] if self.messages and self.messages[0]["role"] == "system" else None
        self.messages.clear()
        if sys_msg:
            self.messages.append(sys_msg)
        self._recent_calls.clear()
        self.steps = 0
        self.total_tokens = 0

    # --- Context management ---
    def _manage_context(self, max_chars: Optional[int] = None) -> None:
        """Sliding window: drop oldest non-system messages when over budget."""
        max_chars = max_chars or settings.AGENT_CONTEXT_WINDOW
        total = sum(len(str(m.get("content", ""))) for m in self.messages)
        while total > max_chars and len(self.messages) > 2:
            # Drop the second message (preserve system + recent)
            dropped = self.messages.pop(1)
            total -= len(str(dropped.get("content", "")))
        # If still too big, drop more from the front
        while total > max_chars and len(self.messages) > 2:
            dropped = self.messages.pop(1)
            total -= len(str(dropped.get("content", "")))

    def _detect_loop(self, call: ToolCall) -> bool:
        """Return True if this call looks like a loop."""
        import json
        try:
            key = (call.name, json.dumps(call.arguments, sort_keys=True))
        except TypeError:
            key = (call.name, str(call.arguments))
        self._recent_calls.append(key)
        if len(self._recent_calls) > self._loop_threshold * 2:
            self._recent_calls = self._recent_calls[-self._loop_threshold * 2:]
        # Count occurrences in recent window
        count = sum(1 for k in self._recent_calls if k == key)
        return count >= self._loop_threshold

    # --- Main entry point ---
    async def run(self, user_input: str) -> AsyncIterator[AgentEvent]:
        """Process a user message, yielding events as the agent works.

        The full conversation history is kept in self.messages; subsequent
        calls to run() will continue the same conversation.
        """
        self.add_user_message(user_input)
        self.start_time = time.time()
        self.steps = 0
        self._recent_calls.clear()

        if self.mode == "chat":
            async for ev in self._run_chat():
                yield ev
            return

        async for ev in self._run_agent():
            yield ev

    async def _run_chat(self) -> AsyncIterator[AgentEvent]:
        """Plain chat: stream tokens, no tools."""
        self._manage_context()
        full = ""
        try:
            async for chunk in self.llm.stream_chat(
                self.messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ):
                if chunk["type"] == "delta":
                    full += chunk["content"]
                    yield AgentEvent("text", {"content": chunk["content"]})
                elif chunk["type"] == "reasoning":
                    yield AgentEvent("reasoning", {"content": chunk["content"]})
                elif chunk["type"] == "error":
                    yield AgentEvent("error", {"message": chunk["message"]})
                    return
        except Exception as e:
            yield AgentEvent("error", {"message": str(e)})
            return

        self.messages.append({"role": "assistant", "content": full})
        elapsed = time.time() - self.start_time
        yield AgentEvent("done", {
            "steps": self.steps,
            "elapsed": round(elapsed, 2),
            "tokens": self.total_tokens,
        })

    async def _run_agent(self) -> AsyncIterator[AgentEvent]:
        """Agent loop: LLM proposes tool calls, we run them, repeat."""
        ctx = make_context(
            workdir=self.workdir,
            auto_approve=self.auto_approve,
            bash_timeout=self.bash_timeout,
        )
        if self.on_bash_request is not None:
            ctx.on_bash_request = self.on_bash_request

        for step in range(1, self.max_steps + 1):
            self.steps = step
            yield AgentEvent("step", {"step": step, "max": self.max_steps})
            self._manage_context()

            # Build tool specs for the prompt (since many models don't have native function calling)
            # We use the text-based <tool_call> format. Some models also support native tools.
            kwargs = {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            # Try native tool calling if all tools have valid schemas
            try:
                tool_specs = [get_tool(n, ctx).to_openai_schema() for n in self.tool_names]
                kwargs["tools"] = tool_specs
                kwargs["tool_choice"] = "auto"
            except Exception:
                pass

            try:
                resp = await self.llm.chat(self.messages, **kwargs)
            except Exception as e:
                yield AgentEvent("error", {"message": f"LLM error: {e}"})
                return

            # Update token count
            if resp.get("usage"):
                self.total_tokens += resp["usage"].get("total_tokens", 0)

            content = resp.get("content") or ""
            tool_calls_native = resp.get("tool_calls") or []

            # Parse tool calls. Prefer native if present, else parse from text.
            calls: List[ToolCall] = []
            display_text = content

            if tool_calls_native:
                for tc in tool_calls_native:
                    fn = tc.get("function") or {}
                    name = fn.get("name", "")
                    args_raw = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                    except Exception:
                        args = {"raw": args_raw}
                    if name:
                        calls.append(ToolCall(name=name, arguments=args, raw=str(tc)))
            else:
                calls, display_text = parse_tool_calls(content, valid_tools=self.tool_names)

            # Stream the visible text to the user
            if display_text:
                yield AgentEvent("text", {"content": display_text})

            # Add the assistant message to history. If we extracted tool calls
            # from the text, store the cleaned text (without tool blocks).
            history_content = content
            if calls and not tool_calls_native:
                history_content = strip_tool_blocks(content)
            if history_content.strip() or calls:
                assistant_msg: Dict[str, Any] = {"role": "assistant", "content": history_content}
                if tool_calls_native:
                    assistant_msg["tool_calls"] = tool_calls_native
                self.messages.append(assistant_msg)

            # No tool calls → done
            if not calls:
                elapsed = time.time() - self.start_time
                yield AgentEvent("done", {
                    "steps": step,
                    "elapsed": round(elapsed, 2),
                    "tokens": self.total_tokens,
                })
                return

            # Execute each call
            looped = False
            for call in calls:
                if self._detect_loop(call):
                    looped = True
                    msg = f"Loop detected: same call '{call.name}' repeated. Stopping."
                    yield AgentEvent("error", {"message": msg})
                    self.messages.append({
                        "role": "user",
                        "content": format_tool_result(call.name, msg, error=True),
                    })
                    break

                yield AgentEvent("tool_call", {
                    "name": call.name,
                    "arguments": call.arguments,
                    "raw": call.raw[:300],
                })
                try:
                    tool = get_tool(call.name, ctx)
                    result = await tool.run(**call.arguments)
                except KeyError:
                    result = type("R", (), {
                        "output": f"error: unknown tool '{call.name}'",
                        "error": True,
                        "metadata": {},
                    })()
                except Exception as e:
                    result = type("R", (), {
                        "output": f"error: tool crashed: {e}",
                        "error": True,
                        "metadata": {},
                    })()

                yield AgentEvent("tool_result", {
                    "name": call.name,
                    "output": result.output,
                    "error": result.error,
                    "metadata": result.metadata,
                })

                # Append tool result to history
                self.messages.append({
                    "role": "tool",
                    "name": call.name,
                    "content": format_tool_result(call.name, result.output, result.error),
                })

            if looped:
                elapsed = time.time() - self.start_time
                yield AgentEvent("done", {
                    "steps": step,
                    "elapsed": round(elapsed, 2),
                    "tokens": self.total_tokens,
                })
                return

        # Hit max_steps
        elapsed = time.time() - self.start_time
        yield AgentEvent("done", {
            "steps": self.max_steps,
            "elapsed": round(elapsed, 2),
            "tokens": self.total_tokens,
            "max_steps_reached": True,
        })
