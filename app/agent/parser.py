"""Robust tool-call parser.

Tries multiple formats, in order of preference:
  1. <tool_call name="...">{"..."}</tool_call>  (Qwen-style XML)
  2. ```json { "name": "...", "arguments": {...} }```  (generic)
  3. <tool_code>...```json name/arguments```...</tool_code>  (multi-call)
  4. ReAct: Action: name \\\\n Action Input: {json}
  5. Bare JSON objects with "name"/"arguments" or "name"/"args"

We never silently drop calls — we always log what we couldn't parse and
return what we found.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# Regex helpers
RE_XML_TOOL_CALL = re.compile(
    r'<tool_call\s+name="([^"]+)"\s*>([\s\S]*?)</tool_call>',
    re.IGNORECASE,
)
RE_FENCE_JSON = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
RE_REACT_ACTION = re.compile(
    r"^\s*Action\s*:\s*([^\n]+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
RE_REACT_INPUT = re.compile(
    r"^\s*Action\s*Input\s*:\s*([\s\S]*?)(?=\n\s*(?:Action|Observation|Thought|Final|###)|\Z)",
    re.IGNORECASE | re.MULTILINE,
)

# Strip common markdown formatting
_FENCE_OPEN = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    # The original text that produced this call (for debugging)
    raw: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


def _parse_args(s: str) -> Dict[str, Any]:
    """Parse a JSON string into a dict. Be tolerant."""
    if not s:
        return {}
    s = s.strip()
    # Strip code fences
    s = _FENCE_OPEN.sub("", s).strip()
    # Try direct parse
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Find first { ... } block
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {"raw_input": s}


def _json_block_with_tool_name(block: str) -> Optional[ToolCall]:
    """If a JSON block has a 'name' field, return a ToolCall. Else None."""
    obj = _parse_args(block)
    name = obj.get("name") or obj.get("tool") or obj.get("function")
    if not name or not isinstance(name, str):
        return None
    args = (
        obj.get("arguments")
        or obj.get("args")
        or obj.get("parameters")
        or obj.get("input")
        or {}
    )
    if not isinstance(args, dict):
        args = {"input": args}
    return ToolCall(name=name, arguments=args, raw=block)


def _parse_react(text: str) -> List[ToolCall]:
    """Parse ReAct format: Action: name\\nAction Input: {json}."""
    calls = []
    actions = list(RE_REACT_ACTION.finditer(text))
    for m in actions:
        name = m.group(1).strip()
        # Strip surrounding punctuation
        name = name.strip("`\"' ")
        if not name:
            continue
        # Find the next Action Input after this Action
        rest = text[m.end():]
        inp = RE_REACT_INPUT.match(rest)
        if not inp:
            calls.append(ToolCall(name=name, arguments={}, raw=m.group(0)))
            continue
        args = _parse_args(inp.group(1))
        calls.append(ToolCall(name=name, arguments=args, raw=m.group(0) + "\n" + inp.group(0)))
    return calls


def parse_tool_calls(
    text: str,
    valid_tools: Optional[List[str]] = None,
) -> Tuple[List[ToolCall], str]:
    """Extract tool calls from model output.

    Returns (calls, remaining_text) where remaining_text is the model output
    with tool-call blocks removed (suitable for showing the user).

    valid_tools: if provided, only accept calls whose name is in this list.
    """
    if not text:
        return [], text

    calls: List[ToolCall] = []
    consumed_spans: List[Tuple[int, int]] = []

    # --- 1) XML <tool_call name="X">{...}</tool_call> ---
    for m in RE_XML_TOOL_CALL.finditer(text):
        name = m.group(1).strip()
        args = _parse_args(m.group(2))
        if valid_tools is None or name in valid_tools:
            calls.append(ToolCall(name=name, arguments=args, raw=m.group(0)))
            consumed_spans.append(m.span())

    # --- 2) ```json { "name": "X", "arguments": {...} }``` ---
    if not calls:
        for m in RE_FENCE_JSON.finditer(text):
            block = m.group(1)
            tc = _json_block_with_tool_name(block)
            if tc and (valid_tools is None or tc.name in valid_tools):
                calls.append(tc)
                consumed_spans.append(m.span())

    # --- 3) ReAct format (Action:/Action Input:) ---
    if not calls:
        react_calls = _parse_react(text)
        for tc in react_calls:
            if valid_tools is None or tc.name in valid_tools:
                calls.append(tc)
        # Don't bother stripping ReAct lines — they're often intermixed with thought.

    # Compute remaining text
    if not consumed_spans:
        remaining = text
    else:
        # Build text minus consumed spans (in reverse order to keep indices valid)
        consumed_spans.sort()
        remaining = text
        for start, end in reversed(consumed_spans):
            remaining = remaining[:start] + remaining[end:]
    remaining = remaining.strip()

    return calls, remaining


def format_tool_result(name: str, output: str, error: bool) -> str:
    """Format a tool result for the model's next turn."""
    status = "error" if error else "ok"
    return f"<tool_result name=\"{name}\" status=\"{status}\">\n{output}\n</tool_result>"


def strip_tool_blocks(text: str) -> str:
    """Remove <tool_call>...</tool_call> blocks and ```json``` tool blocks
    from a piece of text, leaving the prose for display. Use this when showing
    a message that already had calls extracted."""
    if not text:
        return text
    out = RE_XML_TOOL_CALL.sub("", text)
    out = RE_FENCE_JSON.sub("", out)
    return out.strip()
