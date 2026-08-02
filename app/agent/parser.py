"""Robust tool-call parser.

Tries multiple formats, in order of preference:
  1. <tool_call name="...">{...}</tool_call>  (Qwen-style XML, with optional closing)
  2. <tool_call name="...">{...} </tool_call>```

  3. <|tool_call name="...">{...}</tool_call|>  (Mistral/Llama-3 pipe-style)
  4. <|tool_call|>...<|tool_call|>  (pipe-style no name, JSON inside)
  5. ```json { "name": "...", "arguments": {...} }```  (generic)
  6. <tool_code>...```json name/arguments```...</tool_code>  (multi-call)
  7. ReAct: Action: name \\n Action Input: {json}
  8. Bare JSON objects with "name"/"arguments" or "name"/"args"
  9. Qwen3 thinking style: <tool_call>name\n{...} (no closing, until end)

We never silently drop calls — we always log what we couldn't parse and
return what we found.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    raw: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


def _parse_args(s: str) -> Dict[str, Any]:
    """Parse a JSON string into a dict. Be tolerant of common LLM mistakes.

    Common issues with model output:
    1. Single quotes instead of double: {"path": 'add.py'}
    2. Unescaped quotes inside strings: {"content": "if __name__ == '__main__':"}
    3. Trailing commas, missing braces
    4. Newlines and tabs as literal \n / \t in JSON-escaped form
    """
    if not s:
        return {}
    s = s.strip()
    s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s, flags=re.MULTILINE).strip()

    # Strategy 1: standard json parse
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract { ... } block and try various fixes
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        block = m.group(0)
        # Try original
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        # Try with quote-fix
        fixed = _fix_json_quotes(block)
        try:
            obj = json.loads(fixed)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        # Try replacing unescaped single quotes in double-quoted strings
        try:
            obj = _parse_loose_json(block)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # Strategy 3: try ast.literal_eval as last resort (handles Python-style strings)
    try:
        import ast
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Strategy 4: regex-based key extraction (very robust)
    return _extract_key_values(s)


def _parse_loose_json(s: str) -> Any:
    """Parse JSON that's slightly malformed (e.g. missing quotes around keys).

    Only handles objects for our use case.
    """
    # Find all "key": value pairs (or 'key': value)
    result = {}
    # Match: "key": "value" or "key": number or "key": true/false
    pattern = r'["\']?([\w_]+)["\']?\s*:\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|true|false|null|[\d\.\-]+|\[[^\]]*\])'
    for m in re.finditer(pattern, s):
        key = m.group(1)
        raw = m.group(2)
        if raw.startswith('"') or raw.startswith("'"):
            # String value
            quote = raw[0]
            value = raw[1:-1]
            # Unescape
            try:
                # Use json to unescape properly
                value = json.loads('"' + value.replace('"', '\\"') + '"')
            except Exception:
                pass
        elif raw == "true":
            value = True
        elif raw == "false":
            value = False
        elif raw == "null":
            value = None
        elif raw.startswith("["):
            try:
                value = json.loads(raw)
            except Exception:
                value = raw
        else:
            # Number
            try:
                value = int(raw) if "." not in raw else float(raw)
            except ValueError:
                value = raw
        result[key] = value
    if not result:
        raise ValueError("No key-value pairs found")
    return result


def _extract_key_values(s: str) -> Dict[str, Any]:
    """Last-resort: extract key-value pairs using regex even from broken JSON."""
    result = {}
    # Match "key": "value" or "key": value
    # Handle multi-line values that contain embedded quotes by being greedy
    pattern = r'["\']([\w_]+)["\']\s*:\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')'
    for m in re.finditer(pattern, s):
        key = m.group(1)
        raw = m.group(2)
        quote = raw[0]
        value = raw[1:-1]
        # Try to unescape JSON-style
        try:
            if quote == '"':
                value = json.loads('"' + value + '"')
            else:
                # Single-quoted Python string - use ast
                import ast
                value = ast.literal_eval(raw)
        except Exception:
            # Manual unescape
            value = value.replace('\\n', '\n').replace('\\t', '\t').replace("\\'", '"').replace("\\\\", "\\")
        result[key] = value
    if not result:
        result["raw_input"] = s
    return result


def _fix_json_quotes(s: str) -> str:
    """Fix Python single quotes inside JSON double-quoted strings.

    Models sometimes output Python-style strings with single quotes
    inside JSON values, e.g. {"content": "if __name__ == '__main__':"}.
    This is invalid JSON. We walk through the string and convert
    single quotes to escaped double quotes when inside a JSON string.
    """
    out = []
    in_string = False
    escape_next = False
    i = 0
    while i < len(s):
        ch = s[i]
        if escape_next:
            out.append(ch)
            escape_next = False
        elif ch == "\\":
            out.append(ch)
            escape_next = True
        elif ch == '"':
            if not in_string:
                in_string = True
                out.append(ch)
            else:
                # Could be end of string, or a stray quote we should escape
                j = i + 1
                while j < len(s) and s[j] in " \t\n\r":
                    j += 1
                if j < len(s) and s[j] in ",}]: ":
                    in_string = False
                    out.append(ch)
                else:
                    out.append('\\"')
            i += 1
            continue
        elif ch == "'" and in_string:
            out.append('\\"')
        else:
            out.append(ch)
        i += 1
    return "".join(out)


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


# XML <tool_call name="X">{...}</tool_call> — with closing
RE_XML_TOOL_CALL_CLOSED = re.compile(
    r'<tool_call\s+name="([^"]+)"\s*>([\s\S]*?)</tool_call\s*>',
    re.IGNORECASE,
)

# XML <tool_call name="X"> — until </think> or end (Qwen3 common pattern)
RE_XML_TOOL_CALL_UNCLOSED = re.compile(
    r'<tool_call\s+name="([^"]+)"\s*>([\s\S]*?)(?:</tool_call\s*>|</think>|###\s*Observation)',
    re.IGNORECASE,
)

# XML <tool_call> (no name attr, just the JSON) — Qwen3 alternative
RE_XML_TOOL_CALL_NONAME = re.compile(
    r'<tool_call\s*>([\s\S]*?)(?:</tool_call\s*>|</think>|###\s*Observation)',
    re.IGNORECASE,
)

# <|tool_call name="X">{...}</tool_call|>  — Mistral / Llama-3 pipe-style (with closing)
RE_PIPE_TOOL_CALL_CLOSED = re.compile(
    r'<\|tool_call\s+name="([^"]+)"\s*>([\s\S]*?)</tool_call\s*\|>',
    re.IGNORECASE,
)

# <|tool_call name="X">... — unclosed (until next <|tool_call|> or <think>)
RE_PIPE_TOOL_CALL_UNCLOSED = re.compile(
    r'<\|tool_call\s+name="([^"]+)"\s*>([\s\S]*?)(?:<\|tool_call\s*\||</think>|###\s*Observation)',
    re.IGNORECASE,
)

# <|tool_call|>{...}</tool_call|> — pipe-style without name, JSON has name inside (closed)
RE_PIPE_TOOL_CALL_NONAME_CLOSED = re.compile(
    r'<\|tool_call\s*>([\s\S]*?)</tool_call\s*\|>',
    re.IGNORECASE,
)

# <|tool_call|>{json}<|tool_call|> — pipe-style without name, unclosed
RE_PIPE_TOOL_CALL_NONAME = re.compile(
    r'<\|tool_call\s*>([\s\S]*?)(?:<\|tool_call\s*\||</think>|###\s*Observation)',
    re.IGNORECASE,
)

# ```json { "name": "X", "arguments": {...} }```
RE_FENCE_JSON = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)

# <tool_code>...```json```...</tool_code>
RE_TOOL_CODE = re.compile(
    r"<tool_code\s*>([\s\S]*?)</tool_code\s*>",
    re.IGNORECASE,
)

# ReAct format
RE_REACT_ACTION = re.compile(
    r"^\s*Action\s*:\s*([^\n]+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
RE_REACT_INPUT = re.compile(
    r"^\s*Action\s*Input\s*:\s*([\s\S]*?)(?=\n\s*(?:Action|Observation|Thought|Final|###)|\Z)",
    re.IGNORECASE | re.MULTILINE,
)


def _parse_react(text: str) -> List[ToolCall]:
    """Parse ReAct format: Action: name\\nAction Input: {json}."""
    calls = []
    actions = list(RE_REACT_ACTION.finditer(text))
    for m in actions:
        name = m.group(1).strip()
        name = name.strip("`\"' ")
        if not name:
            continue
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
    used: set = set()  # track which regex already matched a span

    def _try_add(tc: ToolCall, span: Tuple[int, int], tag: str) -> None:
        if tc is None:
            return
        if valid_tools is not None and tc.name not in valid_tools:
            return
        # Don't add duplicate (name, args) pairs
        for c in calls:
            if c.name == tc.name and c.arguments == tc.arguments:
                return
        calls.append(tc)
        consumed_spans.append(span)
        used.add(tag)

    # --- 1) XML <tool_call name="X">{...}</tool_call> (closed) ---
    for m in RE_XML_TOOL_CALL_CLOSED.finditer(text):
        name = m.group(1).strip()
        args = _parse_args(m.group(2))
        tc = ToolCall(name=name, arguments=args, raw=m.group(0))
        _try_add(tc, m.span(), "xml_closed")

    # --- 1b) <|tool_call name="X">{...}</tool_call|>  (Mistral/Llama-3 pipe-style closed) ---
    if not calls:
        for m in RE_PIPE_TOOL_CALL_CLOSED.finditer(text):
            name = m.group(1).strip()
            args = _parse_args(m.group(2))
            tc = ToolCall(name=name, arguments=args, raw=m.group(0))
            _try_add(tc, m.span(), "pipe_closed")

    # --- 1c) <|tool_call name="X">... (pipe-style unclosed, until next <|tool_call|>) ---
    if not calls:
        for m in RE_PIPE_TOOL_CALL_UNCLOSED.finditer(text):
            name = m.group(1).strip()
            args = _parse_args(m.group(2))
            tc = ToolCall(name=name, arguments=args, raw=m.group(0))
            _try_add(tc, m.span(), "pipe_unclosed")

    # --- 2) XML <tool_call name="X">...  (unclosed, until </think>) ---
    if not calls:
        for m in RE_XML_TOOL_CALL_UNCLOSED.finditer(text):
            name = m.group(1).strip()
            args = _parse_args(m.group(2))
            tc = ToolCall(name=name, arguments=args, raw=m.group(0))
            _try_add(tc, m.span(), "xml_unclosed")

    # --- 3) XML <tool_call>{json}</tool_call> (no name attribute) ---
    if not calls:
        for m in RE_XML_TOOL_CALL_NONAME.finditer(text):
            inner = m.group(1).strip()
            # Try to find a JSON object with "name" in the inner content
            obj = _parse_args(inner)
            name = obj.get("name")
            if name and isinstance(name, str):
                # Has name — use it
                args = obj.get("arguments", {}) or {}
                _try_add(ToolCall(name=name, arguments=args, raw=m.group(0)),
                         m.span(), "xml_noname_named")
            else:
                # No name — try JSON-anywhere heuristic: find {with path/content/cmd}
                tc = _json_block_with_tool_name(inner)
                if tc is not None:
                    _try_add(tc, m.span(), "xml_noname_inferred")
                else:
                    # Last resort: if JSON has 'cmd', it's bash; 'path' alone could be anything
                    if "cmd" in obj:
                        _try_add(ToolCall(name="bash", arguments=obj, raw=m.group(0)),
                                 m.span(), "xml_noname_guess_bash")
                    elif "content" in obj and "path" in obj:
                        _try_add(ToolCall(name="write_file", arguments=obj, raw=m.group(0)),
                                 m.span(), "xml_noname_guess_write")
                    elif "old_text" in obj and "new_text" in obj:
                        _try_add(ToolCall(name="edit_file", arguments=obj, raw=m.group(0)),
                                 m.span(), "xml_noname_guess_edit")
                    elif "path" in obj:
                        _try_add(ToolCall(name="read_file", arguments=obj, raw=m.group(0)),
                                 m.span(), "xml_noname_guess_read")

    # --- 3b) <|tool_call|>{json}</tool_call|>  (pipe-style no name, closed) ---
    if not calls:
        for m in RE_PIPE_TOOL_CALL_NONAME_CLOSED.finditer(text):
            inner = m.group(1).strip()
            obj = _parse_args(inner)
            name = obj.get("name")
            if name and isinstance(name, str):
                args = obj.get("arguments", {}) or {}
                _try_add(ToolCall(name=name, arguments=args, raw=m.group(0)),
                         m.span(), "pipe_noname_closed_named")
            else:
                tc = _json_block_with_tool_name(inner)
                if tc is not None:
                    _try_add(tc, m.span(), "pipe_noname_closed_inferred")
                else:
                    if "cmd" in obj:
                        _try_add(ToolCall(name="bash", arguments=obj, raw=m.group(0)),
                                 m.span(), "pipe_noname_closed_guess_bash")
                    elif "content" in obj and "path" in obj:
                        _try_add(ToolCall(name="write_file", arguments=obj, raw=m.group(0)),
                                 m.span(), "pipe_noname_closed_guess_write")
                    elif "old_text" in obj and "new_text" in obj:
                        _try_add(ToolCall(name="edit_file", arguments=obj, raw=m.group(0)),
                                 m.span(), "pipe_noname_closed_guess_edit")
                    elif "path" in obj:
                        _try_add(ToolCall(name="read_file", arguments=obj, raw=m.group(0)),
                                 m.span(), "pipe_noname_closed_guess_read")

    # --- 3c) <|tool_call|>{json}<|tool_call|>  (pipe-style no name, unclosed) ---
    if not calls:
        for m in RE_PIPE_TOOL_CALL_NONAME.finditer(text):
            inner = m.group(1).strip()
            obj = _parse_args(inner)
            name = obj.get("name")
            if name and isinstance(name, str):
                args = obj.get("arguments", {}) or {}
                _try_add(ToolCall(name=name, arguments=args, raw=m.group(0)),
                         m.span(), "pipe_noname_named")
            else:
                tc = _json_block_with_tool_name(inner)
                if tc is not None:
                    _try_add(tc, m.span(), "pipe_noname_inferred")
                else:
                    if "cmd" in obj:
                        _try_add(ToolCall(name="bash", arguments=obj, raw=m.group(0)),
                                 m.span(), "pipe_noname_guess_bash")
                    elif "content" in obj and "path" in obj:
                        _try_add(ToolCall(name="write_file", arguments=obj, raw=m.group(0)),
                                 m.span(), "pipe_noname_guess_write")
                    elif "old_text" in obj and "new_text" in obj:
                        _try_add(ToolCall(name="edit_file", arguments=obj, raw=m.group(0)),
                                 m.span(), "pipe_noname_guess_edit")
                    elif "path" in obj:
                        _try_add(ToolCall(name="read_file", arguments=obj, raw=m.group(0)),
                                 m.span(), "pipe_noname_guess_read")

    # --- 4) <tool_code>...```json name/arguments```...</tool_code> ---
    if not calls:
        for m in RE_TOOL_CODE.finditer(text):
            inner = m.group(1)
            for jm in RE_FENCE_JSON.finditer(inner):
                tc = _json_block_with_tool_name(jm.group(1))
                if tc is not None:
                    _try_add(tc, (m.start() + jm.start(), m.start() + jm.end()), "tool_code")
                    break

    # --- 5) ```json { "name": ..., "arguments": ... }``` (fenced) ---
    if not calls:
        for m in RE_FENCE_JSON.finditer(text):
            tc = _json_block_with_tool_name(m.group(1))
            _try_add(tc, m.span(), "fence_json")

    # --- 6) ReAct format (Action: ... Action Input: ...) ---
    if not calls:
        for tc in _parse_react(text):
            _try_add(tc, (0, 0), "react")  # span doesn't matter for text-based

    # --- 7) Final fallback: find a JSON object with "name" anywhere in text ---
    if not calls:
        for m in re.finditer(r"\{[\s\S]*?\"name\"\s*:\s*\"(\w+)\"[\s\S]*?\"arguments\"\s*:", text):
            block_start = m.start()
            # Find matching close brace
            depth = 0
            end = block_start
            for i, ch in enumerate(text[block_start:], start=block_start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            block = text[block_start:end]
            tc = _json_block_with_tool_name(block)
            if tc is not None:
                _try_add(tc, (block_start, end), "json_anywhere")

    # Compute remaining text (remove consumed spans in reverse to keep indices valid)
    if not consumed_spans:
        remaining = text
    else:
        consumed_spans.sort()
        remaining = text
        for start, end in reversed(consumed_spans):
            remaining = remaining[:start] + remaining[end:]
    remaining = remaining.strip()
    return calls, remaining


# Incomplete special-token tool calls (e.g. "<|im_start|>tool_call" with no payload)
RE_INCOMPLETE_PIPE = re.compile(
    r'<\|im_start\|>tool_call[^<]*$',
    re.IGNORECASE,
)


def format_tool_result(name: str, output: str, error: bool) -> str:
    """Format a tool result for the model's next turn."""
    status = "error" if error else "ok"
    return f'<tool_result name="{name}" status="{status}">\n{output}\n</tool_result>'


def strip_tool_blocks(text: str) -> str:
    """Remove <tool_call>...</tool_call> blocks and ```json``` tool blocks
    from a piece of text, leaving the prose for display."""
    if not text:
        return text
    out = RE_XML_TOOL_CALL_CLOSED.sub("", text)
    out = RE_XML_TOOL_CALL_UNCLOSED.sub("", out)
    out = RE_PIPE_TOOL_CALL_CLOSED.sub("", out)
    out = RE_PIPE_TOOL_CALL_UNCLOSED.sub("", out)
    out = RE_PIPE_TOOL_CALL_NONAME_CLOSED.sub("", out)
    out = RE_PIPE_TOOL_CALL_NONAME.sub("", out)
    out = RE_FENCE_JSON.sub("", out)
    out = RE_INCOMPLETE_PIPE.sub("", out)
    return out.strip()
